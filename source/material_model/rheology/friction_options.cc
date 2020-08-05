/*
  Copyright (C) 2019 - 2020 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/material_model/rheology/strain_dependent.h>

#include <deal.II/base/signaling_nan.h>
#include <deal.II/base/parameter_handler.h>
#include <aspect/utilities.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      template <int dim>
      void
      FrictionOptions<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.declare_entry ("Friction options", "default",
                           Patterns::Selection("none|dynamic friction|state dependent friction|default"),
                           "Whether to apply a rate or state dependence of the friction angle. This can "
                           "be used to obtain stick-slip motion to simulate earthquake-like behaviour, "
                           "where short periods of high-velocities are seperated by longer periods without "
                           "movement.  "
                           "\n\n"
                           "\\item ``none'': No rate or state dependence of the friction angle is applied. "
                           "\n\n"
                           "\\item ``dynamic friction'': When dynamic angles of friction are specified, "
                           "the friction angle will be weakened for high strain rates with: "
                           "$\mu = \mu_d + \frac(\mu_s-\mu_d)(1+(\frac(\dot{\epsilon}_{ii})(\dot{\epsilon}_C)))^x$  "
                           "where $\mu_s$ and $\mu_d$ are the friction angle at low and high strain rates, "
                           "respectively. $\dot{\epsilon}_{ii}$ is the second invariant of the strain rate and "
                           "$\dot{\epsilon}_C$ is the characterisitc strain rate where $\mu = (\mu_s+\mu_d)/2$. "
                           "x controls how smooth or step-like the change from $\mu_s$ to $\mu_d$ is. "
                           "The equation is modified after equation 13 in van Dinther et al. 2013. "
                           "\n\n"
                           "\\item ``state dependent friction': A state variable theta is introduced. Method "
                           "is taken from Sobolev and Muldashev 2017. ....."
                           "\n\n"
                           "\\item ``default'': No rate or state dependence of the friction angle is applied. ");

        // Dynamic friction paramters
        prm.declare_entry ("Dynamic characteristic strain rate", "1e-12",
                           Patterns::Double (0),
                           "The characteristic strain rate value, where the angle of friction takes the middle "
                           "between the dynamic and the static angle of friction. When the effective strain rate "
                           "in a cell is very high the dynamic angle of friction is taken, when it is very low "
                           "the static angle of internal friction is chosen. Around the dynamic characteristic "
                           "strain rate, there is a smooth gradient from the static to the dynamic friction "
                           "angle. "
                           "Units: $1/s$.");

        prm.declare_entry ("Dynamic angles of internal friction", "9999",
                           Patterns::List(Patterns::Double(0)),
						   "List of dynamic angles of internal friction, $\\phi$, for background material and compositional "
                           "fields, for a total of N+1 values, where N is the number of compositional fields. "
                           "For a value of zero, in 2D the von Mises criterion is retrieved. ''
                           "Dynamic angles of friction are used for calculation when the effective strain rate in a cell "
                           "is well above the characteristic strain rate. If not specified, the internal angles of "
                           "friction are taken. "
                           "Units: degrees.");
						   
						           for (unsigned int i = 0; i<parameters.dynamic_angles_of_internal_friction.size(); ++i)
          {
            parameters.dynamic_angles_internal_friction[i] *= numbers::PI/180.0;
          }

        prm.declare_entry ("Dynamic friction smoothness exponent", "1",
                           Patterns::List(Patterns::Double(0)),
                           "An exponential factor in the equation for the calculation of the friction angle "
                           "when a static and a dynamic friction angle is specified. A factor =1 is equivalent "
                           "to equation 13 in van Dinther et al., (2013, JGR). A factor between 0 and 1 makes the "
                           "curve of the friction angle vs. the strain rate more smooth, while a factor <1 makes "
                           "the change between static and dynamic friction angle more steplike. "
                           "Units: none.");

        // rate and state parameters
        prm.declare_entry ("Rate and state parameter a", "0",
                           Patterns::List(Patterns::Double(0)),
                           "The rate and state parameter - the rate dependency. Positive (a-b) is velocity"
                           " strengthening, negative (a-b) is velocity weakening. "
                           "Units: none");

        prm.declare_entry ("Rate and state parameter b", "0",
                           Patterns::List(Patterns::Double(0)),
                           "The rate and state parameter - the state dependency. Positive (a-b) is velocity"
                           " strengthening, negative (a-b) is velocity weakening. "
                           "Units: none");

        prm.declare_entry ("Critical slip distance", "0.01",
                           Patterns::List(Patterns::Double(0)),
                           "The critical slip distance in rate and state friction. Used to calculate the state "
                           "variable theta.   "
                           "Units: m");

        prm.declare_entry ("Steady state strain rate", "1e-14",
                           Patterns::List(Patterns::Double(0)),
                           "Arbitrary strain rate at which friction equals the reference friction angle in rate and state friction. "
                           "Units: $1/s$");
      }

      template <int dim>
      void
      FrictionOptions<dim>::parse_parameters (ParameterHandler &prm)
      {
        // Get the number of fields for composition-dependent material properties
        const unsigned int n_fields = this->n_compositional_fields() + 1;

        // number of required compositional fields for full finite strain tensor
        const unsigned int s = Tensor<2,dim>::n_independent_components;

        // Friction dependence parameters
        if (prm.get ("Strain weakening mechanism") == "none")
          weakening_mechanism = none;
        else if (prm.get ("Strain weakening mechanism") == "dynamic friction")
          weakening_mechanism = dynamic_friction;
        else if (prm.get ("Strain weakening mechanism") == "state dependent friction")
          weakening_mechanism = state_dependent_friction;
        else if (prm.get ("Strain weakening mechanism") == "default")
          weakening_mechanism = none;
	  /* would be nice for the future to have an option like rate and state friction with fixed point iteration */
        else
          AssertThrow(false, ExcMessage("Not a valid Strain weakening mechanism!"));

        // Dynamic friction parameters
        dynamic_characteristic_strain_rate = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Dynamic characteristic strain rate"))),
                                                                                     n_fields,
                                                                                     "Dynamic characteristic strain rate");

        dynamic_angles_of_internal_friction = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Dynamic angles of internal friction"))),
                                                                                      n_fields,
                                                                                      "Dynamic angles of internal friction");

        dynamic_friction_smoothness_exponent = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Dynamic friction smoothness exponent"))),
                                                                                       n_fields,
                                                                                       "Dynamic friction smoothness exponent");

        // Rate and state friction parameters
        if (weakening_mechanism == state_dependent_friction)
          {
            AssertThrow(this->introspection().compositional_index_for_name("theta"),
                        ExcMessage("Material model with rate-and-state friction only works "
                                   "if there is a compositional field that is called theta."));
          }


        rate_and_state_parameter_a = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Rate and state parameter a"))),
                                                                             n_fields,
                                                                             "Rate and state parameter a");

        rate_and_state_parameter_b = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Rate and state parameter b"))),
                                                                             n_fields,
                                                                             "Rate and state parameter b");

        critical_slip_distance = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Critical slip distance"))),
                                                                         n_fields,
                                                                         "Critical slip distance");

        steady_state_strain_rate = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Steady state strain rate"))),
                                                                           n_fields,
                                                                           "Steady state strain rate");
      }


      template <int dim>
      std::array<double, 3>
      FrictionOptions<dim>::
      compute_dependent_friction_angle(const unsigned int j,
                                       const std::vector<double> &composition
                                       const MaterialModel::MaterialModelInputs<dim> &in,
                                       const int i,
                                       const double min_strain_rate,
                                       MaterialModel::MaterialModelOutputs<dim> &out) const
      /* what is j what is i? They came from different functions before. Do I need both? */
      {
        /* do I have to declare current_friction here???? */

        // compute current_edot_ii
        const double current_edot_ii = compute_edot_ii ()

                                       switch (weakening_mechanism)
          {
            case none:
            {
              break;
            }
            case dynamic_friction:
            {
              // The dynamic characteristic strain rate is used to see if dynamic or static angle of internal friction should be used.
              // This is done as in the material_model dynamic_friction which is based on equation 13 in van Dinther et al., (2013, JGR). 
              // const double mu  = mu_d[i] + (mu_s[i] - mu_d[i]) / ( (1 + strain_rate_dev_inv2/reference_strain_rate) );
              // which is the following using the variables in this material_model. Although here the dynamic friction coefficient
			  // is directly specified. The coefficient of friction is the tangent of the internal angle of friction. 
			  // Furthermore a smoothness coefficient is added, which influences if the friction vs strain rate curve is rather 
			  // step-like or more gradual.
              const double mu = std::tan(dynamic_angles_of_internal_friction[j])
                                              + (std::tan(drucker_prager_parameters.angles_internal_friction[j])   //    do I need the weakening factors here? I guess not:  * weakening_factors[1]
                                                 - std::tan(dynamic_angles_of_internal_friction[j]))
                                              / (1 + std::pow((current_edot_ii / dynamic_characteristic_strain_rate[j]),
                                                              dynamic_friction_smoothness_exponent));
			   const double current_friction = std::atan (mu);
              break;
            }
            case state_dependent_friction:
            {
              //cellsize is needed for theta and the friction angle
              double cellsize = 1;
              if (in.current_cell.state() == IteratorState::valid)
                {
                  cellsize = in.current_cell->extent_in_direction(0);
                }

              // calculate the state variable theta
              // theta_old loads theta from previous time step

              const unsigned int theta_position_tmp = this->introspection().compositional_index_for_name("theta");
              double theta_old = composition[theta_position_tmp];
              // equation (7) from Sobolev and Muldashev 2017
              const double theta = critical_slip_distance[j] / cellsize / current_edot_ii + (theta_old - critical_slip_distance[j]
                                                                                             / cellsize/current_edot_ii) * exp( - (current_edot_ii * this->get_timestep())
                                                                                                 / critical_slip_distance[j] * cellsize);

              // calculate effective friction according to equation (4) in Sobolev and Muldashev 2017;
              // effective friction id calculated by multiplying the friction coefficient with 0.03 = (1-p_f/sigma_n)
              // their equation is for friction coefficient, while ASPECT takes friction angle in RAD, so conversion with tan/atan()
              const double current_friction = atan(0.03*(tan(drucker_prager_parameters.angles_internal_friction[j])
                                                         + rate_and_state_parameter_a[j] * log(current_edot_ii * cellsize
                                                             / steady_state_strain_rate[j]) + rate_and_state_parameter_b[j]
                                                         * log(theta * steady_state_strain_rate[j] / critical_slip_distance[j])));

              break;
            }
            default:
            {
              AssertThrow(false, ExcNotImplemented());
              break;
            }
          }
        /*
                std::array<double, 3> weakening_factors = {brittle_weakening.first,brittle_weakening.second,viscous_weakening};
        */
        return current_friction;
      }

      template <int dim>
      void
      FrictionOptions<dim>::
      fill_reaction_outputs (const MaterialModel::MaterialModelInputs<dim> &in,
                             const int i,
                             const double min_strain_rate,
                             const bool plastic_yielding,
                             MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        /* do I need this???? */
      }

      template <int dim>
      ComponentMask
      FrictionOptions<dim>::
      get_volumetric_composition_mask() const
      {
        // Store which components to exclude during the volume fraction computation.
        /* copied from visco_plastic: check how to get that information!!!! */
        ComponentMask composition_mask = strain_rheology.get_strain_composition_mask();

        if (weakening_mechanism == state_dependent_friction)
          {
            // this is the compositional field used for theta in rate-and-state friction
            int theta_position_tmp = this->introspection().compositional_index_for_name("theta");
            composition_mask.set(theta_position_tmp,false);
          }

        return composition_mask;
      }


      template <int dim>
      void
      FrictionOptions<dim>::
      compute_edot_ii (const MaterialModel::MaterialModelInputs<dim> &in,
                       const int i,
                       const double min_strain_rate,
                       MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        if (this->simulator_is_past_initialization() && this->get_timestep_number() > 0 && in.requests_property(MaterialProperties::reaction_terms) && in.current_cell.state() == IteratorState::valid)
          {
            for (unsigned int q=0; q < in.n_evaluation_points(); ++q)
              {
                const bool use_reference_strainrate = (this->get_timestep_number() == 0) &&
                                                      ((in.strain_rate[q]).norm() <= std::numeric_limits<double>::min());
                double edot_ii;
                if (use_reference_strainrate)
                  edot_ii = ref_strain_rate;
                else
                  edot_ii = std::max(std::sqrt(std::fabs(second_invariant(deviator(in.strain_rate[q])))),
                                     min_strain_rate);

                double current_edot_ii = numbers::signaling_nan<double>();
                SymmetricTensor<2,dim> stress_old = numbers::signaling_nan<SymmetricTensor<2,dim>>();

                if (use_elasticity == false)
                  {
                    current_edot_ii = edot_ii;
                  }
                else
                  {
                    for (unsigned int j=0; j < SymmetricTensor<2,dim>::n_independent_components; ++j)
                      {
                        stress_old[SymmetricTensor<2,dim>::unrolled_to_component_indices(j)] = in.composition[q][j];
                        const std::vector<double> &elastic_shear_moduli = elastic_rheology.get_elastic_shear_moduli();
                        if (use_reference_strainrate == true)
                          current_edot_ii = ref_strain_rate;
                        else
                          {
                            const double viscoelastic_strain_rate_invariant = elastic_rheology.calculate_viscoelastic_strain_rate(in.strain_rate[q],
                                                                              stress_old,
                                                                              elastic_shear_moduli[j]);
                            current_edot_ii = std::max(viscoelastic_strain_rate_invariant,
                                                       min_strain_rate);
                          }
                      }
                    current_edot_ii /= 2.;
                  }
              }
          }
        return current_edot_ii;
      }


      template <int dim>
      void
      FrictionOptions<dim>::
      compute_theta_reaction_terms(const MaterialModel::MaterialModelInputs<dim> &in,
                                   MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        //cellsize is needed for theta and the friction angle
        double cellsize = 1;
        if (in.current_cell.state() == IteratorState::valid)
          {
            cellsize = in.current_cell->extent_in_direction(0);
          }

        // compute current_edot_ii
        const double current_edot_ii = compute_edot_ii ()


                                       const unsigned int theta_position_tmp = this->introspection().compositional_index_for_name("theta");
        double theta_old = in.composition[q][theta_position_tmp];
        // equation (7) from Sobolev and Muldashev 2017. Though here I had to add  "- theta_old"
        // because I need the change in theta for reaction_terms
        const double theta_increment = critical_slip_distance[theta_position_tmp+1] /cellsize/current_edot_ii +
                                       (theta_old - critical_slip_distance[theta_position_tmp+1]
                                        /cellsize/current_edot_ii)*exp(-(current_edot_ii*this->get_timestep())
                                                                       /critical_slip_distance[theta_position_tmp+1]*cellsize) - theta_old;
        out.reaction_terms[q][theta_position_tmp] = theta_increment;

        /*
        if (in.current_cell.state() == IteratorState::valid && this->get_timestep_number() > 0 && in.requests_property(MaterialProperties::reaction_terms))
          {
            // We need the velocity gradient for the finite strain (they are not
            // in material model inputs), so we get them from the finite element.
            std::vector<Point<dim> > quadrature_positions(in.n_evaluation_points());
            for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
              quadrature_positions[i] = this->get_mapping().transform_real_to_unit_cell(in.current_cell, in.position[i]);

            FEValues<dim> fe_values (this->get_mapping(),
                                     this->get_fe(),
                                     Quadrature<dim>(quadrature_positions),
                                     update_gradients);

            std::vector<Tensor<2,dim> > velocity_gradients (quadrature_positions.size(), Tensor<2,dim>());

            fe_values.reinit (in.current_cell);
            fe_values[this->introspection().extractors.velocities].get_function_gradients (this->get_solution(),
                                                                                           velocity_gradients);

            // Assign the strain components to the compositional fields reaction terms.
            // If there are too many fields, we simply fill only the first fields with the
            // existing strain tensor components.

            for (unsigned int q=0; q < in.n_evaluation_points(); ++q)
              {
                if (in.current_cell.state() == IteratorState::valid && weakening_mechanism == finite_strain_tensor
                    && this->get_timestep_number() > 0 && in.requests_property(MaterialProperties::reaction_terms))

                  {
                    // Convert the compositional fields into the tensor quantity they represent.
                    Tensor<2,dim> strain;
                    const unsigned int n_first = this->introspection().compositional_index_for_name("s11");
                    for (unsigned int i = n_first; i < n_first + Tensor<2,dim>::n_independent_components ; ++i)
                      {
                        strain[Tensor<2,dim>::unrolled_to_component_indices(i)] = in.composition[q][i];
                      }

                    // Compute the strain accumulated in this timestep.
                    const Tensor<2,dim> strain_increment = this->get_timestep() * (velocity_gradients[q] * strain);

                    // Output the strain increment component-wise to its respective compositional field's reaction terms.
                    for (unsigned int i = n_first; i < n_first + Tensor<2,dim>::n_independent_components ; ++i)
                      {
                        out.reaction_terms[q][i] = strain_increment[Tensor<2,dim>::unrolled_to_component_indices(i)];
                      }
                  }
              }
          }
        */
      }



      template <int dim>
      WeakeningMechanism
      FrictionOptions<dim>::
      get_weakening_mechanism() const
      {
        return weakening_mechanism;
      }

    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
  namespace Rheology \
  { \
    template class FrictionOptions<dim>; \
  }

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
