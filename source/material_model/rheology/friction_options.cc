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


#include <aspect/material_model/rheology/friction_options.h>

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
        prm.declare_entry ("Friction dependence mechanism", "default",
                           Patterns::Selection("none|dynamic friction|state dependent friction|default"),
                           "Whether to apply a rate and/or state dependence of the friction angle. This can "
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
                           "\\item ``state dependent friction'': A state variable theta is introduced. Method "
                           "is taken from Sobolev and Muldashev 2017. ....."
                           "\n\n"
                           "\\item ``default'': No rate or state dependence of the friction angle is applied. ");

        // Plasticity parameters
        /*should I do this or just read in the internal anlges of friction directly? */
        /*drucker_prager_parameters = drucker_prager_plasticity.parse_parameters(this->n_compositional_fields()+1,
                                                                               prm);*/

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
                           "For a value of zero, in 2D the von Mises criterion is retrieved. "
                           "Dynamic angles of friction are used for calculation when the effective strain rate in a cell "
                           "is well above the characteristic strain rate. If not specified, the internal angles of "
                           "friction are taken. "
                           "Units: degrees.");

        prm.declare_entry ("Dynamic friction smoothness exponent", "1",
                           Patterns::Double (0),
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

        // Friction dependence parameters
        if (prm.get ("Friction dependence mechanism") == "none")
          friction_dependence_mechanism = independent;
        else if (prm.get ("Friction dependence mechanism") == "dynamic friction")
          friction_dependence_mechanism = dynamic_friction;
        else if (prm.get ("Friction dependence mechanism") == "state dependent friction")
          friction_dependence_mechanism = state_dependent_friction;
        else if (prm.get ("Friction dependence mechanism") == "default")
          friction_dependence_mechanism = independent;
        /* would be nice for the future to have an option like rate and state friction with fixed point iteration */
        else
          AssertThrow(false, ExcMessage("Not a valid friction dependence option!"));

        // Dynamic friction parameters
        dynamic_characteristic_strain_rate = prm.get_double("Dynamic characteristic strain rate");

        if (prm.get ("Dynamic angles of internal friction") == "9999")
          {
            // If not specified, the internal angles of friction are used, so there is no dynamic friction in the model
            dynamic_angles_of_internal_friction = drucker_prager_parameters.angles_internal_friction;
          }
        else
          {
            dynamic_angles_of_internal_friction = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Dynamic angles of internal friction"))),
                                                                                          n_fields,
                                                                                          "Dynamic angles of internal friction");

            // Convert angles from degrees to radians
            for (unsigned int i = 0; i<dynamic_angles_of_internal_friction.size(); ++i)
              {
                if (dynamic_angles_of_internal_friction[i] > 90)
                  {
                    AssertThrow(false, ExcMessage("Dynamic angles of friction must be <= 90 degrees"));
                  }
                else
                  {
                    dynamic_angles_of_internal_friction[i] *= numbers::PI/180.0;
                  }
              }
          }

        dynamic_friction_smoothness_exponent = prm.get_double("Dynamic friction smoothness exponent");

        // Rate and state friction parameters
        if (friction_dependence_mechanism == state_dependent_friction)
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

        critical_slip_distance = prm.get_double("Critical slip distance");

        steady_state_strain_rate = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Steady state strain rate"))),
                                                                           n_fields,
                                                                           "Steady state strain rate");
      }

      template <int dim>
      double
      FrictionOptions<dim>::
      compute_dependent_friction_angle(const double current_edot_ii,
                                       const unsigned int j,  // volume fraction
                                       const std::vector<double> &composition,
                                       typename DoFHandler<dim>::active_cell_iterator current_cell,
                                       double current_friction) const
      {

        switch (friction_dependence_mechanism)
          {
            case independent:
            {
              /* I guess here I need to say that current friction is simply the internal angle of friction. If not it will return 0.0 as its value?  */
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
                                + (std::tan(current_friction)
                                   - std::tan(dynamic_angles_of_internal_friction[j]))
                                / (1 + std::pow((current_edot_ii / dynamic_characteristic_strain_rate),
                                                dynamic_friction_smoothness_exponent));
              current_friction = std::atan (mu);
              break;
            }
            case state_dependent_friction:
            {
              //cellsize is needed for theta and the friction angle
              double cellsize = 1;
              if (current_cell.state() == IteratorState::valid)
                {
                  cellsize = current_cell->extent_in_direction(0);
                  // calculate the state variable theta
                  // theta_old loads theta from previous time step

                  const unsigned int theta_position_tmp = this->introspection().compositional_index_for_name("theta");
                  double theta_old = composition[theta_position_tmp];
                  // equation (7) from Sobolev and Muldashev 2017
                  const double theta = critical_slip_distance / cellsize /
                                       current_edot_ii + (theta_old - critical_slip_distance
                                                          / cellsize / current_edot_ii) * exp( - (current_edot_ii * this->get_timestep())
                                                                                               / critical_slip_distance * cellsize);

                  // calculate effective friction according to equation (4) in Sobolev and Muldashev 2017;
                  // effective friction id calculated by multiplying the friction coefficient with 0.03 = (1-p_f/sigma_n)
                  // their equation is for friction coefficient, while ASPECT takes friction angle in RAD, so conversion with tan/atan()
                  current_friction = atan(0.03*(tan(current_friction)
                                                + rate_and_state_parameter_a[j] * log(current_edot_ii * cellsize
                                                                                      / steady_state_strain_rate[j]) + rate_and_state_parameter_b[j]
                                                * log(theta * steady_state_strain_rate[j] / critical_slip_distance)));
                  break;
                }
              else
                {
                  break;
                }
            }
            default:
            {
              AssertThrow(false, ExcNotImplemented());
              break;
            }
          }
        return current_friction;
      }

      template <int dim>
      ComponentMask
      FrictionOptions<dim>::
      get_theta_composition_mask(ComponentMask composition_mask) const
      {
        // Store which components to exclude during the volume fraction computation.

        if (friction_dependence_mechanism == state_dependent_friction)
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
      compute_theta_reaction_terms(const MaterialModel::MaterialModelInputs<dim> &in,
                                   const std::vector<double> &volume_fractions,
                                   const double min_strain_rate,
                                   const double ref_strain_rate,
                                   bool use_elasticity,
                                   bool use_reference_strainrate,
                                   const std::vector<double> &elastic_shear_moduli,
                                   const double dte,
                                   MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        //cellsize is needed for theta and the friction angle
        double cellsize = 1;
        if (in.current_cell.state() == IteratorState::valid)
          {
            cellsize = in.current_cell->extent_in_direction(0);
          }

        if (this->simulator_is_past_initialization() && this->get_timestep_number() > 0 && in.requests_property(MaterialProperties::reaction_terms) && in.current_cell.state() == IteratorState::valid)
          {
            for (unsigned int j=0; j < volume_fractions.size(); ++j)
              {
                for (unsigned int q=0; q < in.n_evaluation_points(); ++q)
                  {
                    const double current_edot_ii =
                      MaterialUtilities::compute_current_edot_ii (in.composition[q], ref_strain_rate,
                                                                  min_strain_rate, in.strain_rate[q], elastic_shear_moduli[j], use_elasticity,
                                                                  use_reference_strainrate, dte);
                    const unsigned int theta_position_tmp = this->introspection().compositional_index_for_name("theta");
                    double theta_old = in.composition[q][theta_position_tmp];
                    // equation (7) from Sobolev and Muldashev 2017. Though here I had to add  "- theta_old"
                    // because I need the change in theta for reaction_terms
                    const double theta_increment = critical_slip_distance /cellsize/current_edot_ii +
                                                   (theta_old - critical_slip_distance
                                                    /cellsize/current_edot_ii)*exp(-(current_edot_ii*this->get_timestep())
                                                                                   /critical_slip_distance*cellsize) - theta_old;
                    out.reaction_terms[q][theta_position_tmp] = theta_increment;
                  }
              }
          }
      }

      template <int dim>
      FrictionDependenceMechanism
      FrictionOptions<dim>::
      get_friction_dependence_mechanism() const
      {
        return friction_dependence_mechanism;
      }

      template <int dim>
      bool
      FrictionOptions<dim>::
      get_theta_in_use() const
      {
        bool theta_in_use =false;
        if (get_friction_dependence_mechanism() == state_dependent_friction)
          theta_in_use =true;

        return theta_in_use;
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

