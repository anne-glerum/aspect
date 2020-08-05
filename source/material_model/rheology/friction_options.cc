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
                             "Dynamic angles of friction which are taken when the effective strain rate in a cell "
                             "is well above the characteristic strain rate. If not specified, the internal angles of "
                             "friction are taken. "
                             "Units: degrees.");

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
                                       const std::vector<double> &composition) const
      {
        double viscous_weakening = 1.0;
        std::pair<double, double> brittle_weakening (1.0, 1.0);

        switch (weakening_mechanism)
          {
            case none:
            {
              break;
            }
            case dynamic_friction:
            {
              // Calculate second invariant of left stretching tensor "L"
			  /*
              Tensor<2,dim> strain;
              for (unsigned int q = 0; q < Tensor<2,dim>::n_independent_components ; ++q)
                strain[Tensor<2,dim>::unrolled_to_component_indices(q)] = composition[q];
              const SymmetricTensor<2,dim> L = symmetrize( strain * transpose(strain) );

              const double strain_ii = std::fabs(second_invariant(L));
              brittle_weakening = calculate_plastic_weakening(strain_ii, j);
			  */
              viscous_weakening = calculate_viscous_weakening(strain_ii, j);
              break;
            }
            case state_dependent_friction:
            {
				/*
              const unsigned int total_strain_index = this->introspection().compositional_index_for_name("total_strain");
              brittle_weakening = calculate_plastic_weakening(composition[total_strain_index], j);
              viscous_weakening = calculate_viscous_weakening(composition[total_strain_index], j);
			  */
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

        return weakening_factors;
*/
      }


      template <int dim>
      std::pair<double, double>
      StrainDependent<dim>::
      calculate_plastic_weakening(const double strain_ii,
                                  const unsigned int j) const
      {
        // Constrain the second strain invariant of the previous timestep by the strain interval
        const double cut_off_strain_ii = std::max(std::min(strain_ii,end_plastic_strain_weakening_intervals[j]),start_plastic_strain_weakening_intervals[j]);

        // Linear strain weakening of cohesion and internal friction angle between specified strain values
        const double strain_fraction = (cut_off_strain_ii - start_plastic_strain_weakening_intervals[j]) /
                                       (start_plastic_strain_weakening_intervals[j] - end_plastic_strain_weakening_intervals[j]);

        const double weakening_cohesion = 1. + (1. - cohesion_strain_weakening_factors[j]) * strain_fraction;
        const double weakening_friction = 1. + (1. - friction_strain_weakening_factors[j]) * strain_fraction;

        return std::make_pair (weakening_cohesion, weakening_friction);
      }


      template <int dim>
      double
      StrainDependent<dim>::
      calculate_viscous_weakening(const double strain_ii,
                                  const unsigned int j) const
      {
        // Constrain the second strain invariant of the previous timestep by the strain interval
        const double cut_off_strain_ii = std::max(std::min(strain_ii,end_viscous_strain_weakening_intervals[j]),start_viscous_strain_weakening_intervals[j]);

        // Linear strain weakening of the viscous flow law prefactors between specified strain values
        const double strain_fraction = (cut_off_strain_ii - start_viscous_strain_weakening_intervals[j]) /
                                       (start_viscous_strain_weakening_intervals[j] - end_viscous_strain_weakening_intervals[j]);
        return 1. + ( 1. - viscous_strain_weakening_factors[j] ) * strain_fraction;
      }

      template <int dim>
      void
      StrainDependent<dim>::
      fill_reaction_outputs (const MaterialModel::MaterialModelInputs<dim> &in,
                             const int i,
                             const double min_strain_rate,
                             const bool plastic_yielding,
                             MaterialModel::MaterialModelOutputs<dim> &out) const
      {

        // If strain weakening is used, overwrite the first reaction term,
        // which represents the second invariant of the (plastic) strain tensor.
        // If plastic strain is tracked (so not the total strain), only overwrite
        // when plastically yielding.
        // If viscous strain is also tracked, overwrite the second reaction term as well.
        // Calculate changes in strain and update the reaction terms
        if  (this->simulator_is_past_initialization() && this->get_timestep_number() > 0 && in.requests_property(MaterialProperties::reaction_terms))
          {
            const double edot_ii = std::max(sqrt(std::fabs(second_invariant(deviator(in.strain_rate[i])))),min_strain_rate);
            const double e_ii = edot_ii*this->get_timestep();
            if (weakening_mechanism == plastic_weakening_with_plastic_strain_only && plastic_yielding == true)
              out.reaction_terms[i][this->introspection().compositional_index_for_name("plastic_strain")] = e_ii;
            if (weakening_mechanism == viscous_weakening_with_viscous_strain_only && plastic_yielding == false)
              out.reaction_terms[i][this->introspection().compositional_index_for_name("viscous_strain")] = e_ii;
            if (weakening_mechanism == total_strain || weakening_mechanism == plastic_weakening_with_total_strain_only)
              out.reaction_terms[i][this->introspection().compositional_index_for_name("total_strain")] = e_ii;
            if (weakening_mechanism == plastic_weakening_with_plastic_strain_and_viscous_weakening_with_viscous_strain)
              {
                if (plastic_yielding == true)
                  out.reaction_terms[i][this->introspection().compositional_index_for_name("plastic_strain")] = e_ii;
                else
                  out.reaction_terms[i][this->introspection().compositional_index_for_name("viscous_strain")] = e_ii;
              }
            if (this->introspection().compositional_name_exists("noninitial_plastic_strain") && plastic_yielding == true)
              out.reaction_terms[i][this->introspection().compositional_index_for_name("noninitial_plastic_strain")] = e_ii;
          }
      }


      template <int dim>
      void
      StrainDependent<dim>::
      compute_finite_strain_reaction_terms(const MaterialModel::MaterialModelInputs<dim> &in,
                                           MaterialModel::MaterialModelOutputs<dim> &out) const
      {

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
      }

      template <int dim>
      ComponentMask
      StrainDependent<dim>::
      get_strain_composition_mask() const
      {

        // Store which components to exclude during volume fraction computation.
        ComponentMask strain_mask(this->n_compositional_fields(),true);

        if (weakening_mechanism != none)
          {
            if (weakening_mechanism == plastic_weakening_with_plastic_strain_only || weakening_mechanism == plastic_weakening_with_plastic_strain_and_viscous_weakening_with_viscous_strain)
              strain_mask.set(this->introspection().compositional_index_for_name("plastic_strain"),false);

            if (weakening_mechanism == viscous_weakening_with_viscous_strain_only || weakening_mechanism == plastic_weakening_with_plastic_strain_and_viscous_weakening_with_viscous_strain)
              strain_mask.set(this->introspection().compositional_index_for_name("viscous_strain"),false);

            if (weakening_mechanism == total_strain || weakening_mechanism == plastic_weakening_with_total_strain_only)
              strain_mask.set(this->introspection().compositional_index_for_name("total_strain"),false);

            if (weakening_mechanism == finite_strain_tensor)
              {
                const unsigned int n_start = this->introspection().compositional_index_for_name("s11");
                for (unsigned int i = n_start; i < n_start + Tensor<2,dim>::n_independent_components ; ++i)
                  strain_mask.set(i,false);
              }
          }

        if (this->introspection().compositional_name_exists("noninitial_plastic_strain"))
          strain_mask.set(this->introspection().compositional_index_for_name("noninitial_plastic_strain"),false);

        return strain_mask;
      }

      template <int dim>
      WeakeningMechanism
      StrainDependent<dim>::
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
    template class StrainDependent<dim>; \
  }

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}

