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
#include <aspect/geometry_model/interface.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      std::vector<std::string> make_friction_additional_outputs_names()
      {
        std::vector<std::string> names;
        names.emplace_back("RSF_a");
        names.emplace_back("RSF_b");
        names.emplace_back("RSF_L");
        names.emplace_back("edot_ii");
        return names;
      }
    }



    template <int dim>
    FrictionAdditionalOutputs<dim>::FrictionAdditionalOutputs (const unsigned int n_points)
      :
      NamedAdditionalMaterialOutputs<dim>(make_friction_additional_outputs_names()),
      RSF_a(n_points, numbers::signaling_nan<double>()),
      RSF_b(n_points, numbers::signaling_nan<double>()),
      RSF_L(n_points, numbers::signaling_nan<double>()),
      edot_ii(n_points, numbers::signaling_nan<double>())
    {}



    template <int dim>
    std::vector<double>
    FrictionAdditionalOutputs<dim>::get_nth_output(const unsigned int idx) const
    {
      (void)idx; // suppress warning in release mode
      /*
      TODO:
      I copied this from elasticity.cc. However I get this warning during 'make':
      /home/hecken/code/aspect/source/material_model/rheology/friction_options.cc: In member function ‘std::vector<double, std::allocator<double> > aspect::MaterialModel::FrictionAdditionalOutputs<dim>::get_nth_output(unsigned int) const [with int dim = 3]’:
      /home/hecken/code/aspect/source/material_model/rheology/friction_options.cc:82:5: warning: control reaches end of non-void function [-Wreturn-type]
      */
      AssertIndexRange (idx, 4);
      switch (idx)
        {
          case 0:
            return RSF_a;

          case 1:
            return RSF_b;

          case 2:
            return RSF_L;

          case 3:
            return edot_ii;
        }
      // We will never get here, so just return something
      return RSF_a;
    }



    namespace Rheology
    {
      template <int dim>
      double
      FrictionOptions<dim>::
      compute_dependent_friction_angle(const double current_edot_ii,
                                       const unsigned int j,  // j is from a for-loop over volume_fractions.size()
                                       const std::vector<double> &composition,  // these are the compositional fields not volume_fractions
                                       typename DoFHandler<dim>::active_cell_iterator current_cell,
                                       double current_friction,
                                       const Point<dim> &position,
                                       const double current_cohesion,
                                       const double pressure_for_plasticity,
                                       const double max_yield_stress,
                                       const double current_stress,
                                       const double min_strain_rate) const
      {

        switch (friction_dependence_mechanism)
          {
            case independent:
            {
              break;
            }
            case dynamic_friction:
            {
              const double effective_friction_factor = get_effective_friction_factor(position);
              // The dynamic characteristic strain rate is used to see what value between dynamic and static angle of internal friction should be used.
              // This is done as in the material model dynamic friction which is based on Equation (13) in van Dinther et al., (2013, JGR). Although here
              // the dynamic friction coefficient is directly specified. Furthermore a smoothness exponent X is added, which determines whether the
              // friction vs strain rate curve is rather step-like or more gradual.
              // mu  = mu_d + (mu_s - mu_d) / ( (1 + strain_rate_dev_inv2/dynamic_characteristic_strain_rate)^X );
              // Angles of friction are used in radians within ASPECT. The coefficient of friction is the tangent of the internal angle of friction.
              const double mu = (1 - effective_friction_factor)
                                * (std::tan(dynamic_angles_of_internal_friction[j])
                                   + (std::tan(current_friction) - std::tan(dynamic_angles_of_internal_friction[j]))
                                   / (1. + std::pow((current_edot_ii / dynamic_characteristic_strain_rate),
                                                    dynamic_friction_smoothness_exponent)));
              current_friction = std::atan (mu);
              Assert((mu < 1) && (0 < current_friction <=1.6), ExcMessage(
                       "Something is wrong with the tan/atan conversion of friction coefficient to friction angle in RAD."));
              break;
            }
            case steady_state_rate_and_state_dependent_friction:
            {
              double cellsize = 1.;
              if (current_cell.state() == IteratorState::valid)
                {
                  cellsize = current_cell->extent_in_direction(0);
                  // Get the values for a and b and the critcal slip distance L
                  const double rate_and_state_parameter_a = calculate_depth_dependent_a_and_b(position,j).first;
                  const double rate_and_state_parameter_b = calculate_depth_dependent_a_and_b(position,j).second;
                  const double effective_friction_factor = get_effective_friction_factor(position);

                  const double mu = (1 - effective_friction_factor)
                                    * (std::tan(current_friction)
                                       + (rate_and_state_parameter_a - rate_and_state_parameter_b)
                                       * std::log(steady_state_velocity
                                                  / (quasi_static_strain_rate * cellsize)));
                  current_friction = std::atan (mu);
                }
              break;
            }
            // default is for case rate_and_state_dependent_friction with the other rate-and-state variations as if statements
            default:
            {
              // Cellsize is needed for theta and the friction angle
              // For now, the used cells are non-deforming squares, so the edge length in the
              // x-direction is representative of the cell size.
              // TODO: In case of mesh deformation this length might not be representative
              double cellsize = 1.;
              if (current_cell.state() == IteratorState::valid)
                {
                  cellsize = current_cell->extent_in_direction(0);

                  // Get the values for a and b and the critcal slip distance L
                  double rate_and_state_parameter_a = calculate_depth_dependent_a_and_b(position,j).first;
                  const double rate_and_state_parameter_b = calculate_depth_dependent_a_and_b(position,j).second;
                  double critical_slip_distance = get_critical_slip_distance(position,j);
                  const double effective_friction_factor = get_effective_friction_factor(position);

                  // theta_old is taken from the current compositional field theta
                  double theta_old = composition[theta_composition_index];
                  if(theta_old < 0)
                    std::cout << "got a negative old theta before computing friction" << std::endl;
                  theta_old = std::max(theta_old,1e-50);

                  // if we do not assume always yielding, then theta should not be updated
                  // with the entire strain rate, but only with min strain rate
                  // ToDo: Should I do the same to compute the friction angle? would be tricky, bc it is computed before the yielding thing...

                  double current_edot_ii_for_theta = current_edot_ii;
                  const double yield_stress = drucker_prager_plasticity.compute_yield_stress(current_cohesion,
                                                                                             current_friction,
                                                                                             pressure_for_plasticity,
                                                                                             max_yield_stress);
                  if (current_stress < yield_stress)
                    current_edot_ii_for_theta = min_strain_rate;

                  // Calculate the state variable theta according to Equation (7) from Sobolev and Muldashev (2017)
                  const double theta = compute_theta(theta_old, current_edot_ii_for_theta, cellsize, critical_slip_distance);

                  //std::cout << "theta = "<<theta<<" - theta_old = "<< theta_old<<std::endl;

                  if (friction_dependence_mechanism == slip_rate_dependent_rate_and_state_dependent_friction)
                    {
                      // compute slip-rate dependence following Equations 8 and 9 in \\cite{im_slip-rate-dependent_2020}
                      rate_and_state_parameter_a += slope_s_for_a * std::log10((ref_v_for_a + current_edot_ii * cellsize)/ref_v_for_a);

                      critical_slip_distance += slope_s_for_L * std::log10((ref_v_for_L + current_edot_ii * cellsize)/ref_v_for_L);
                    }

                  double mu = 0;
                  if (friction_dependence_mechanism == regularized_rate_and_state_dependent_friction)
                    {
                      // As we divide by RSF parameter a here, current_friction will be nan when a is zero
                      // Not modifying the friction angle in the case of a (and b) zero follows the classical
                      // logarithmic formulation of rate-and-state friction.
                      if (rate_and_state_parameter_a > 0)
                        {
                          // Calculate regularized rate and state friction (e.g. Herrendörfer 2018) with
                          // mu = a sinh^{-1}[V/(2V_0)exp((mu_0 + b ln(V_0 theta/L))/a)]
                          // Their equation is for friction coefficient.
                          // As we use strain-rates and current_edot_ii instead of velocities, these
                          // are multiplied by the cellsize.
                          // Effective friction is explained below for the other friction option.
                          mu = (1 - effective_friction_factor)
                               * (rate_and_state_parameter_a
                                  * std::asinh(current_edot_ii / (2.0 * quasi_static_strain_rate)
                                               * std::exp((std::tan(current_friction)
                                                           + rate_and_state_parameter_b
                                                           * std::log((theta * quasi_static_strain_rate * cellsize) / critical_slip_distance))
                                                          / rate_and_state_parameter_a)));
                        }
                      else
                        mu = (1 - effective_friction_factor)
                             * (std::tan(current_friction) + rate_and_state_parameter_b
                                * std::log((theta * quasi_static_strain_rate  * cellsize) / critical_slip_distance));
                    }
                  else
                    {
                      // Calculate effective friction according to Equation (4) in Sobolev and Muldashev (2017):
                      // mu = mu_st + a ln(V/V_st) + b ln((theta V_st)/L)
                      // Their equation is for friction coefficient.
                      // Effective friction is calculated by multiplying the friction coefficient with the
                      // effective_friction_factor to account for effects of pore fluid pressure:
                      // mu = mu(1-p_f/sigma_n) = mu*, with (1-p_f/sigma_n) = 0.03 for subduction zones.
                      //const double current_friction_old = current_friction; // also only for chasing negative friction
                      mu = (1 - effective_friction_factor)
                           * (std::tan(current_friction)
                              + rate_and_state_parameter_a
                              * std::log(current_edot_ii / quasi_static_strain_rate)
                              + rate_and_state_parameter_b
                              * std::log((theta * quasi_static_strain_rate  * cellsize) / critical_slip_distance));

                      /* TODO: from Sobolev and Muldashev appendix.
                      if (friction_dependence_mechanism == rate_and_state_dependent_friction_plus_linear_slip_weakening)
                      {
                      const double deltamustartvonD  ;
                      mu -= deltamustartvonD;
                      }*/
                    }
                  // All equations for the different friction options are for friction coefficient, while
                  // ASPECT takes friction angle in radians, so conversion with tan/atan().
                  current_friction = std::atan (mu);
                  const std::array<double,dim> coords = this->get_geometry_model().cartesian_to_other_coordinates(position, coordinate_system_RSF).get_coordinates();
                  Assert((mu < 1) && (0 < current_friction <=1.6), ExcMessage(
                           "Something is wrong with the tan/atan conversion of friction coefficient to friction angle in RAD"));
                  AssertThrow((std::isinf(mu) || numbers::is_nan(mu)) == false, ExcMessage(
                                "Your friction coefficient becomes nan or inf. Please check all your friction parameters. In case of "
                                "rate-and-state like friction, don't forget to check on a,b, and the critical slip distance, or theta."
                                "\n a is: "+  Utilities::to_string(rate_and_state_parameter_a) + ", b is: "
                                + Utilities::to_string(rate_and_state_parameter_b)+ ", L is: " +  Utilities::to_string(critical_slip_distance) +
                                ",\n effecitve friction factor is: "+ Utilities::to_string(effective_friction_factor) +
                                ",\n friction angle [RAD] is: "+ Utilities::to_string(current_friction)+", friction coeff is: "+Utilities::to_string(mu)+
                                ",\n theta is: "+ Utilities::to_string(theta)+", current edot_ii is: "
                                + Utilities::to_string(current_edot_ii)+ ".\n The position is:\n dir 0 = "+ Utilities::to_string(coords[0])+
                                "\n dir 1 = "+ Utilities::to_string(coords[1])+ "\n dir 2 = "+ Utilities::to_string(coords[2])));

                  // chasing the origin of negative friction angles
                  if (theta <= 0)
                    {
                      std::cout << "Theta is zero/negative: " << theta << " at time " << this->get_time() << std::endl;
                      std::cout << "Previous theta was: " << theta_old << std::endl;
                      std::cout << "current edot ii * cellsize is " << current_edot_ii *cellsize << std::endl;
                      std::cout << "current edot ii is " << current_edot_ii<< std::endl;
                      std::cout << "the friction coeff at this time is: " << tan(current_friction) << " and the friction angle in RAD is " << current_friction << std::endl;
                      std::cout << "the friction angle in degree is: " << current_friction*180/3.1516 << std::endl;
                      //AssertThrow(false, ExcMessage("Theta negative."));
                    }/*
                  if (current_friction <= 0)
                    {
                      std::cout << "current_friction is zero/negative!"<<std::endl;
                      std::cout << "Theta is " << theta << " at time " << this->get_time() << std::endl;
                      std::cout << "Previous theta was: " << theta_old << std::endl;
                      std::cout << "current edot ii * cellsize is " << current_edot_ii *cellsize << std::endl;
                      std::cout << "current edot ii is " << current_edot_ii<< std::endl;
                      std::cout << "the friction coeff at this time is: " << tan(current_friction) << " and the friction angle in RAD is " << current_friction << std::endl;
                      std::cout << "the friction angle in degree is: " << current_friction*180/3.1516 << std::endl;
                      std::cout << "the effective friction factor is: " << effective_friction_factor[j] << std::endl;
                      std::cout << " the current_friction before the RSF is [RAD] " << current_friction_old << std::endl;
                      std::cout << "a is: "<<rate_and_state_parameter_a<< " and b is: "<< rate_and_state_parameter_b << std::endl;
                    }*/
                }
              break;
            }
          }
        // A negative friction angle, that does not make sense and will get rate-and-state friction
        // into trouble, so return some very small value
        if (friction_dependence_mechanism != regularized_rate_and_state_dependent_friction)
          current_friction = std::max(current_friction, 1e-30);
        return current_friction;
      }



      template <int dim>
      double
      FrictionOptions<dim>::
      compute_theta(double theta_old,
                    const double current_edot_ii,
                    const double cellsize,
                    const double critical_slip_distance) const
      {
        // this is a trial to check if it prevents current_theta from being negative if old_theta is limited to >=0
        theta_old = std::max(theta_old,1e-50);
        // Equation (7) from Sobolev and Muldashev (2017):
        // theta_{n+1} = L/V_{n+1} + (theta_n - L/V_{n+1})*exp(-(V_{n+1}dt)/L)
        // This is obtained from Equation (5): dtheta/dt = 1 - (theta V)/L
        // by integration using the assumption that velocities are constant at any time step.
        double current_theta = critical_slip_distance / ( cellsize * current_edot_ii ) +
                               (theta_old - critical_slip_distance / ( cellsize * current_edot_ii))
                               * std::exp( - (current_edot_ii * cellsize) * this->get_timestep() / critical_slip_distance);

        // TODO: make dt the min theta?
        // Theta needs a cutoff towards zero and negative values, because these
        // values physically do not make sense but can occur as theta is advected
        // as a material field. A zero or negative value for theta also leads to nan
        // values for friction.
        if (current_theta < 0)
           std::cout << "got theta negative" << std::endl;
        //current_theta = std::max(current_theta, 1e-50);
        return current_theta;
      }



      template <int dim>
      void
      FrictionOptions<dim>::
      compute_theta_reaction_terms(const int q,
                                   const std::vector<double> &volume_fractions,
                                   const MaterialModel::MaterialModelInputs<dim> &in,
                                   const double min_strain_rate,
                                   const double ref_strain_rate,
                                   const bool use_elasticity,
                                   const bool use_reference_strainrate,
                                   const double &average_elastic_shear_moduli,
                                   const double dte,
                                   MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        if (this->simulator_is_past_initialization() && this->get_timestep_number() > 0
            && in.requests_property(MaterialProperties::reaction_terms)
            && in.current_cell.state() == IteratorState::valid)
          {
            // q is from a for-loop through n_evaluation_points
            const double current_edot_ii =
              MaterialUtilities::compute_current_edot_ii (in.composition[q], ref_strain_rate,
                                                          min_strain_rate, in.strain_rate[q],
                                                          average_elastic_shear_moduli, use_elasticity,
                                                          use_reference_strainrate, dte);

            double theta_old = in.composition[q][theta_composition_index];
            if(theta_old < 0)
               std::cout << "got a negative old theta in theta reaction terms" << std::endl;
            theta_old = std::max(theta_old,1e-50);
            double current_theta = 0;
            double critical_slip_distance = 0.0;

            if (friction_dependence_mechanism == steady_state_rate_and_state_dependent_friction)
              {
                for (unsigned int j=0; j < volume_fractions.size(); ++j)
                  critical_slip_distance += volume_fractions[j] * get_critical_slip_distance(in.position[q], j);
                current_theta += critical_slip_distance / steady_state_velocity;
              }
            else
              {
                for (unsigned int j=0; j < volume_fractions.size(); ++j)
                  {
                    critical_slip_distance += volume_fractions[j] * get_critical_slip_distance(in.position[q], j);
                    // theta is also computed within the loop to keep it as similar as possible to the theta used
                    // within compute_dependent_friction which is also called from within a loop over volume_fractions.size()
                    current_theta += volume_fractions[j] * compute_theta(theta_old, current_edot_ii,
                                                                         in.current_cell->extent_in_direction(0), critical_slip_distance);
                  }
              }

            if (current_theta == 1e-50)
              std::cout << "got a cut theta in theta reaction terms" << std::endl;

            // prevent negative theta values in reaction terms by cutting the increment additionally to current_theta
            double theta_increment = current_theta - theta_old;
            if (theta_old + theta_increment < 1e-50)
              theta_increment = 1e-50 - theta_old;

            out.reaction_terms[q][theta_composition_index] = theta_increment;
          }
      }



      template <int dim>
      std::pair<double,double>
      FrictionOptions<dim>::calculate_depth_dependent_a_and_b(const Point<dim> &position, const int j) const
      {
        Utilities::NaturalCoordinate<dim> point =
          this->get_geometry_model().cartesian_to_other_coordinates(position, coordinate_system_RSF);

        const double rate_and_state_parameter_a =
          rate_and_state_parameter_a_function->value(Utilities::convert_array_to_point<dim>(point.get_coordinates()),j);
        const double rate_and_state_parameter_b =
          rate_and_state_parameter_b_function->value(Utilities::convert_array_to_point<dim>(point.get_coordinates()),j);

        return std::pair<double,double>(rate_and_state_parameter_a,
                                        rate_and_state_parameter_b);
      }



      template <int dim>
      double
      FrictionOptions<dim>::get_critical_slip_distance(const Point<dim> &position, const int j) const
      {
        Utilities::NaturalCoordinate<dim> point =
          this->get_geometry_model().cartesian_to_other_coordinates(position, coordinate_system_RSF);

        const double critical_slip_distance =
          critical_slip_distance_function->value(Utilities::convert_array_to_point<dim>(point.get_coordinates()),j);

        AssertThrow(critical_slip_distance > 0, ExcMessage("Critical slip distance in a rate-and-state material must be > 0."));

        return critical_slip_distance;
      }



      template <int dim>
      double
      FrictionOptions<dim>::get_effective_friction_factor(const Point<dim> &position) const
      {
        Utilities::NaturalCoordinate<dim> point =
          this->get_geometry_model().cartesian_to_other_coordinates(position, coordinate_system_RSF);

        const double effective_friction_factor =
          effective_friction_factor_function.value(Utilities::convert_array_to_point<dim>(point.get_coordinates()));

        AssertThrow((effective_friction_factor >= 0) && (effective_friction_factor <1), ExcMessage("Effective friction factor must be < 1 and >= 0, "
                                                              "because anything else will cause negative or zero friction coefficients."));

        return effective_friction_factor;
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
      use_theta() const
      {
        bool use_theta = false;
        const FrictionDependenceMechanism mechanism = get_friction_dependence_mechanism();
        // steady_state_rate_and_state_dependent_friction does not actually use theta, but
        // it also comes from the RSF framework and can potentially be used to compute a
        // longterm model that will then be restarted with real RSF. Real RSF needs the
        // compositional field theta, so theta needs to be in the longterm model too. However,
        // if steady-state RSF is not listed here to use theta, theta will not be excluded
        // from volume_fractions, except for it being zero. However, for real RSF theta is
        // not allowed to be 0. So probably it is easiest if steady-state RSF also sets the
        // use_theta flag to true.
        if ((mechanism == rate_and_state_dependent_friction)
            || (mechanism == rate_and_state_dependent_friction_plus_linear_slip_weakening)
            || (mechanism == regularized_rate_and_state_dependent_friction)
            || (mechanism == slip_rate_dependent_rate_and_state_dependent_friction)
            || (mechanism == steady_state_rate_and_state_dependent_friction))
          use_theta = true;

        return use_theta;
      }



      template <int dim>
      void
      FrictionOptions<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.declare_entry ("Friction dependence mechanism", "none",
                           Patterns::Selection("none|dynamic friction|rate and state dependent friction|"
                                               "rate and state dependent friction plus linear slip weakening|"
                                               "slip rate dependent rate and state dependent friction|"
                                               "regularized rate and state dependent friction|"
                                               "steady state rate and state dependent friction"),
                           "Whether to apply a rate or rate-and-state dependence of the friction angle. This can "
                           "be used to obtain stick-slip motion to simulate earthquake-like behaviour, "
                           "where short periods of high-velocities are separated by longer periods without "
                           "movement."
                           "\n\n"
                           "\\item ``none'': No rate or state dependence of the friction angle is applied. "
                           "\n\n"
                           "\\item ``dynamic friction'': The friction angle is rate dependent."
                           "When dynamic angles of friction are specified, "
                           "the friction angle will be weakened for high strain rates with: "
                           "$\\mu = \\mu_d + \\frac(\\mu_s-\\mu_d)(1+(\\frac(\\dot{\\epsilon}_{ii})(\\dot{\\epsilon}_C)))^x$  "
                           "where $\\mu_s$ and $\\mu_d$ are the friction angle at low and high strain rates, "
                           "respectively. $\\dot{\\epsilon}_{ii}$ is the second invariant of the strain rate and "
                           "$\\dot{\\epsilon}_C$ is the characterisitc strain rate where $\\mu = (\\mu_s+\\mu_d)/2$. "
                           "x controls how smooth or step-like the change from $\\mu_s$ to $\\mu_d$ is. "
                           "The equation is modified after Equation (13) in \\cite{van_dinther_seismic_2013}. "
                           "$\\mu_s$ and $\\mu_d$ can be specified by setting 'Angles of internal friction' and "
                           "'Dynamic angles of internal friction', respectively."
                           "\n\n"
                           "\\item ``rate and state dependent friction'': A state variable theta is introduced "
                           "and the friction angle is calculated using classic aging rate-and-state friction by "
                           "Ruina (1983) as described by Equations (4--7) in \\cite{sobolev_modeling_2017}:\n"
                           "$\\mu = \\mu_{st} + a \\cdot ln\\big( \\frac{V}{V_{st}} \\big) + b \\cdot ln\\big( \\frac{\\theta V_{st}}{L} \\big)$,\n"
                           "$\\frac{d\\theta}{dt} = 1-\\frac{\\theta V}{L} $.\n"
                           "Assuming that velocities are constant at any time step, this can be analytically integrated: \n"
                           "$\\theta_{n+1} = \\frac{L}{V_{n+1}} + \\big(\\theta_n - \\frac{L}{V_{n+1}}\\big)*exp\\big(-\\frac{V_{n+1}\\Delta t}{L}\\big)$.\n"
                           "Pore fluid pressure can be taken into account by specifying the 'Effective friction "
                           "factor', which uses $\\mu* = \\mu\\big(1-\\frac{P_f}{\\sigma_n} \\big)$. "
                           "In ASPECT the state variable is confined to values > 1e-50: if it becomes $<1e-50$ during the computation "
                           "it is set to 1e-50. The same applies to the friction angle which is set to 1e-30 if smaller than that. "
                           "The term $a \\cdot ln\\big( \\frac{V}{V_{st}} \\big)$ is often referred to as the instantaneous 'viscosity-like' "
                           "direct effect, and the rate-and-state parameter a describes its magnitude. "
                           "The term $b \\cdot ln\\big( \\frac{\\theta V_{st}}{L} \\big)$ is referred to as the evolution effect as it is described "
                           "by the evolving state variable $\\theta$. The magnitude of the evolution effect is determined by the "
                           "rate-and-state parameter b. See \\cite{herrendorfer_invariant_2018} for a comprehensive explanation of all parameters. "
                           "Reasonable values for a and b are 0.01 and 0.015, respectively, see \\cite{sobolev_modeling_2017}. "
                           "The critical slip distance L in rate-and-state friction is used to calculate the "
                           "state variable theta using the aging law: $\\frac{d\\theta}{dt}=1-\\frac{\\theta V}{L}$. "
                           "At steady state: $\\theta = \\frac{L}{V} $. The critical slip distance "
                           "is often interpreted as the sliding distance required to renew "
                           "the contact population as it determines the critical nucleation size with: "
                           "$h*=\\frac{2}{\\pi}\\frac{\\mu b L}{(b-a)^2 \\sigma_n}$, where $\\sigma_n$ is the normal stress on the fault. "
                           "Laboratory values of the critical slip distance are on the "
                           "order of microns. For geodynamic modelling \\cite{sobolev_modeling_2017} set this parameter "
                           "to 1--10 cm. In the SCEC-SEAS benchmark initiative \\citep{erickson_community_2020} they use 4 and 8 mm. "
                           "This parameter should be changed when the level of mesh refinement de- or increases. "
                           "I has the Unit: \\si{\\meter}."
                           "The parameters a, b, and the critical slip distance L are specified as functions in a separate subsections."
                           "The $V_{st}$ is the quasi-static strain-rate at which the friction coefficient will match the reference "
                           "friction coefficient $\\mu_{st}$ which is defined through the 'angle of internal friction' parameter. "
                           "\n\n"
                           "\\item ``rate and state dependent friction plus linear slip weakening'': ToDo: all, this is an empty model atm. "
                           "Method taken from \\cite{sobolev_modeling_2017}. The friction coefficient is computed as "
                           "$\\mu = \\mu_{st} + a \\cdot ln\\big( \\frac{V}{V_{st}} \\big) + b \\cdot ln\\big( \\frac{\\theta V_{st}}{L} \\big) - \\Delta \\mu(D)$"
                           "where D is the slip at point in fault at the first timestep of earthquake. "
                           "\n\n"
                           "\\item ``slip rate dependent rate and state dependent friction'': The rate-and-state "
                           "parameters and the critical slip distance L are made slip-rate dependent. The friction "
                           "coefficient is computed as in 'rate and state dependent friction'. But a and L are not "
                           "constant, but are computed as follows, see \\citep{Im_im_slip-rate-dependent_2020} for details."
                           "$a(V) = a_0 + s_a log_{10}\\left(\\frac{V_a+V}{V_a}\\right)$ and "
                           "$L(V) = L_0 + s_L log_{10}\\left(\\frac{V_L+V}{V_L}\\right)$."
                           "So a and L have a log linear dependence on velocity with slopes od $s_a$ and $s_L$. "
                           "Parameters in their paper have the following values: "
                           "$L_0=10\\mu m$, $s_L=60\\mu m$, $V_L=100\\mu m/s$ and "
                           "$a_0=0.005$, $s_a=0.0003$, $V_a=100\\mu m/s$. "
                           "In ASPECT the initial values $a_0$ and $L_0$ are the rate-and-state friction parameters "
                           "indicated in 'Critical slip distance function' and 'Rate and state parameter a function'."
                           "\n\n"
                           "\\item ``regularized rate and state dependent friction'': The friction coefficient is computed using: "
                           "$\\mu = a\\cdot sinh^{-1}\\left[\\frac{V}{2V_0}exp\\left(\\frac{\\mu_0+b\\cdot ln(V_0\\theta/L)}{a}\\right)\\right]$ "
                           "This is a high velocity approximation and regularized version of the classic rate-and-state friction. "
                           "This formulation overcomes the problem of ill-posedness and the possibility of negative friction for "
                           "$V<<V_0$. It is for example used in \\cite{herrendorfer_invariant_2018}. "
                           "\n\n"
                           "\\item ``steady state rate and state dependent friction'': friction "
                           "is computed as the steady-state friction coefficient in rate-and-state friction: "
                           "$\\mu_{ss} =\\mu_{st}}+(a-b)ln(V_{ss}/V_{st}})$"
                           "This friction is reached when state evolves toward a steady state $\\theta_{ss} = L/V_{ss}$"
                           "at constant slip velocities. The velocity $V_{ss}$ can be specified with the parameter "
                           "'steady state velocity for RSF'. Note that not the actual velocities in the model are taken "
                           "to determine the friction angle, but a constant $V_{ss}$. If actual model velocities would "
                           "be used for $V_{ss}$, this would be another rate-dependent friction formulation. This friction "
                           "formulation is thought to be useful for models where the friction angle should dependent on a and b, "
                           "but not on rate nor state, e.g. in a model run with long timesteps to acquire the initial conditions "
                           "for a rate-and-state model. Even though a state variable theta is not used to compute the friction "
                           "angle, a compositional field 'theta' must be provided to ensure a smooth restart with one of the "
                           "rate-and-state friction formulations. Theta is computed to equal $\\theta_{ss}$.");

        // Dynamic friction paramters
        prm.declare_entry ("Dynamic characteristic strain rate", "1e-12",
                           Patterns::Double (0),
                           "The characteristic strain rate value, where the angle of friction takes the middle "
                           "between the dynamic and the static angle of friction. When the effective strain rate "
                           "in a cell is very high, the dynamic angle of friction is taken, when it is very low "
                           "the static angle of internal friction is chosen. Around the dynamic characteristic "
                           "strain rate, there is a smooth gradient from the static to the dynamic friction "
                           "angle. "
                           "Units: \\si{\\per\\second}.");

        prm.declare_entry ("Dynamic angles of internal friction", "2",
                           Patterns::List(Patterns::Double(0)),
                           "List of dynamic angles of internal friction, $\\phi$, for background material and compositional "
                           "fields, for a total of N+1 values, where N is the number of compositional fields. "
                           "Dynamic angles of friction are used as the current friction angle when the effective "
                           "strain rate in a cell is well above the characteristic strain rate. "
                           "Units: \\si{\\degree}.");

        prm.declare_entry ("Dynamic friction smoothness exponent", "1",
                           Patterns::Double (0),
                           "An exponential factor in the equation for the calculation of the friction angle "
                           "when a static and a dynamic friction angle are specified. A factor of 1 returns the equation "
                           "to Equation (13) in \\cite{van_dinther_seismic_2013}. A factor between 0 and 1 makes the "
                           "curve of the friction angle vs. the strain rate more smooth, while a factor $>$ 1 makes "
                           "the change between static and dynamic friction angle more steplike. "
                           "Units: none.");

        /**
         * The functions for the rate-and-state parameters a and b, and the critical
         * slip distance use the same coordinate system. They can be declared in dependence
         * of depth, cartesian coordinates or spherical coordinates. Note that the order
         * of spherical coordinates is r,phi,theta and not r,theta,phi, since
         * this allows for dimension independent expressions.
         */
        prm.declare_entry ("Coordinate system for RSF parameters", "cartesian",
                           Patterns::Selection ("cartesian|spherical|depth"),
                           "A selection that determines the assumed coordinate "
                           "system for the function variables. Allowed values "
                           "are `cartesian', `spherical', and `depth'. `spherical' coordinates "
                           "are interpreted as r,phi or r,phi,theta in 2D/3D "
                           "respectively with theta being the polar angle. `depth' "
                           "will create a function, in which only the first "
                           "parameter is non-zero, which is interpreted to "
                           "be the depth of the point.");

        prm.enter_subsection("Rate and state parameter a function");
        {
          Functions::ParsedFunction<dim>::declare_parameters(prm,1);
        }
        prm.leave_subsection();
        prm.enter_subsection("Rate and state parameter b function");
        {
          Functions::ParsedFunction<dim>::declare_parameters(prm,1);
        }
        prm.leave_subsection();

        prm.declare_entry ("Effective friction factor", "1",
                           Patterns::List(Patterns::Double(0)),
                           "A number that is multiplied with the coefficient of friction to take into "
                           "account the influence of pore fluid pressure. This makes the friction "
                           "coefficient an effective friction coefficient as in \\cite{sobolev_modeling_2017}. "
                           "Units: none.");

        prm.declare_entry ("Effective normal stress on fault", "1",
                           Patterns::List(Patterns::Double(0)),
                           "This is a scalar value for the effective normal stress on a fault. It "
                           "replaces the solution-dependent normal stress in Tresca friction formulation. "
                           "This is for example used in \\cite{erickson_community_2020} and "
                           "\\cite{pipping_variational_2015} in simple rate-and-state friction models. "
                           "This parameter only becomes effective when the yield mechanism tresca "
                           "is specified in the input file. "
                           "Units: Pa.");

        prm.enter_subsection("Critical slip distance function");
        {
          Functions::ParsedFunction<dim>::declare_parameters(prm,1);
        }
        prm.leave_subsection();

        prm.enter_subsection("Effective friction factor function");
        {
          Functions::ParsedFunction<dim>::declare_parameters(prm,1);
        }
        prm.leave_subsection();

        prm.declare_entry ("Quasi static strain rate", "1e-14",
                           Patterns::Double (0),
                           "The quasi static or reference strain rate used in rate and state friction. It is an "
                           "arbitrary strain rate at which friction equals the reference friction angle. "
                           "This happens when slip rate (which is represented in ASPECT as strain rate * cell size) "
                           "enters a steady state. Friction at this steady state is defined as: "
                           "$\\mu = \\mu_{st} = \\mu_0 + (a-b)ln\\big( \\frac{V}{V_0} \\big). "
                           "It should not be confused with the characteristic strain rate in dynamic friction. "
                           "Units: \\si{\\per\\second}.");

        prm.declare_entry ("Steady state velocity for RSF", "1.75e-2",
                           Patterns::Double (0),
                           "This is velocity $V_{st}$ used in 'steady state rate and state dependent friction'. "
                           "In the rate-and-state friction framework, when the velocity remains constant "
                           "friction will cease to change and evolve to a steady-state that depends on a and b. "
                           "This is the input parameter to choose the desired velocity, that does not actually "
                           "is the velocity in the model but is used to determine the friction angle based on the "
                           "equation in 'steady state rate and state dependent friction'. So this velocity is "
                           "assumed to have remained constant over a long period such that the friction angle "
                           "evolved to a steady-state."
                           "Units: \\si{\\meter\\per\\second}.");

        prm.declare_entry ("Use radiation damping", "false",
                           Patterns::Bool (),
                           "Whether to include radiation damping or not. Radiation damping adds the term "
                           "$-\\etaV = \\frac{G}{2c_s}$ to the yield stress, $G$ is the elastic shear "
                           "modulus, $c_s$ the shear wave speed and V the slip rate \\citep{rice_spatio-temporal_1993}. "
                           "Radiation damping prevents velocities from increasing to infinity at the small time steps of "
                           "earthquakes. It therewith assures that the governing equations continue to have a solution "
                           "during earthquakelike episodes. Unlike an inertial term it cannot be used to model rupture "
                           "propagation as it approximates seismic waves as energy outflow only. ");

        prm.declare_entry ("Cut edot_ii after radiation damping", "true",
                           Patterns::Bool (),
                           "Whether to cut edot_ii after applying radiation damping or not. A parameter for debugging my issues. ");

        prm.declare_entry ("Use always yielding", "false",
                           Patterns::Bool (),
                           "Whether to assume always yielding for the compositional field 'fault'. In the "
                           "rate-and-state friction framework material is assumed to always be at yield. ");

        /* TODO:
        prm.declare_entry ("Friction state variable law", "aging law",
                          Patterns::Selection ("aging law|slip law"),
                          "A selection that determines the law used for the evolution of the state variable "
                          "$\\theta$ in rate-and-state friction. When the aging law (also called Dietrich law "
                          "or slowness law) is chosen, the state variable is computed with "
                          "$\\dot{\\theta}=1-\\frac{V\\theta}{D_c}$ while the slip law (or Ruina law) follows "
                          "$\\dot{\\theta}=-\\frac{V\\theta}{D_c}ln\\frac{V\\theta}{D_c}$, where $D_c$ is the "
                          "critical slip distance and V is velocity. The aging law indicates that state "
                          "increases when $V=0$, whereas the slip law requires slip to occur: no evolution in "
                          "state occurs unless $V\\neq0$ \\citep[read more in ]{scholz_mechanics_2019}."); */

        // Parameters for slip rate dependent rate and state friction
        prm.declare_entry ("Reference velocity for rate and state parameter a", "0.005",
                           Patterns::Double (0),
                           "The reference velocity used to modify the initial value for rate and state "
                           "parameter a in case of slip rate dependence. "
                           "Units: \\si{\\meter\\per\\second}.");

        prm.declare_entry ("Reference velocity for critical slip distance L", "0.01",
                           Patterns::Double (0),
                           "The reference velocity used to modify the initial value for rate and state "
                           "parameter L, the critical slip distance in case of slip rate dependence. "
                           "Units: \\si{\\meter\\per\\second}.");

        prm.declare_entry ("Slope of log dependence for rate and state parameter a", "3e-4",
                           Patterns::Double (0),
                           "Slope for the log linear slip rate dependence of rate and state parameter a."
                           "Units: \\si{\\meter}.");

        prm.declare_entry ("Slope of log dependence for critical slip distance L", "6e-5",
                           Patterns::Double (0),
                           "Slope for the log linear slip rate dependence of the critical slip distance. "
                           "Units: \\si{\\meter}.");
      }



      template <int dim>
      void
      FrictionOptions<dim>::parse_parameters (ParameterHandler &prm)
      {
        // Get the number of fields for composition-dependent material properties
        // including the background field.
        const unsigned int n_fields = this->n_compositional_fields() + 1;

        // Friction dependence parameters
        if (prm.get ("Friction dependence mechanism") == "none")
          friction_dependence_mechanism = independent;
        else if (prm.get ("Friction dependence mechanism") == "dynamic friction")
          friction_dependence_mechanism = dynamic_friction;
        else if (prm.get ("Friction dependence mechanism") == "rate and state dependent friction")
          friction_dependence_mechanism = rate_and_state_dependent_friction;
        else if (prm.get ("Friction dependence mechanism") == "rate and state dependent friction plus linear slip weakening")
          friction_dependence_mechanism = rate_and_state_dependent_friction_plus_linear_slip_weakening;
        else if (prm.get ("Friction dependence mechanism") == "slip rate dependent rate and state dependent friction")
          friction_dependence_mechanism = slip_rate_dependent_rate_and_state_dependent_friction;
        else if (prm.get ("Friction dependence mechanism") == "regularized rate and state dependent friction")
          friction_dependence_mechanism = regularized_rate_and_state_dependent_friction;
        else if (prm.get ("Friction dependence mechanism") == "steady state rate and state dependent friction")
          friction_dependence_mechanism = steady_state_rate_and_state_dependent_friction;
        else
          AssertThrow(false, ExcMessage("Not a valid friction dependence option!"));

        if (use_theta())
          {
            // Currently, it only makes sense to use a state variable when the nonlinear solver
            // scheme does a single Advection iteration and at minimum one Stokes iteration, as
            // the state variable is implemented as a material field. More
            // than one nonlinear Advection iteration will produce an unrealistic values
            // for the state variable theta.
            AssertThrow((this->get_parameters().nonlinear_solver ==
                         Parameters<dim>::NonlinearSolver::single_Advection_single_Stokes
                         ||
                         this->get_parameters().nonlinear_solver ==
                         Parameters<dim>::NonlinearSolver::single_Advection_iterated_Stokes
                         ||
                         this->get_parameters().nonlinear_solver ==
                         Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes
                         ||
                         this->get_parameters().nonlinear_solver ==
                         Parameters<dim>::NonlinearSolver::single_Advection_iterated_defect_correction_Stokes),
                        ExcMessage("The rate and state friction will only work with the nonlinear "
                                   "solver schemes 'single Advection, single Stokes' and "
                                   "'single Advection, iterated Stokes', 'single Advection, "
                                   "iterated defect correction Stokes', single advection, iterated Newton Stokes'"));
          }

        // Dynamic friction parameters
        dynamic_characteristic_strain_rate = prm.get_double("Dynamic characteristic strain rate");

        dynamic_angles_of_internal_friction = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Dynamic angles of internal friction"))),
                                                                                      n_fields,
                                                                                      "Dynamic angles of internal friction");
        // Convert angles from degrees to radians
        for (unsigned int i = 0; i<dynamic_angles_of_internal_friction.size(); ++i)
          {
            AssertThrow(dynamic_angles_of_internal_friction[i] <= 90, ExcMessage("Dynamic angles of friction must be <= 90 degrees"));
            dynamic_angles_of_internal_friction[i] *= numbers::PI/180.0;
          }

        dynamic_friction_smoothness_exponent = prm.get_double("Dynamic friction smoothness exponent");

        // Rate and state friction parameters
        prm.enter_subsection("Effective friction factor function");
        try
          {
            effective_friction_factor_function.parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "FunctionParser failed to parse\n"
                      << "\t Effective friction factor function\n"
                      << "with expression \n"
                      << "\t' " << prm.get("Function expression") << "'";
            throw;
          }
        prm.leave_subsection();

        prm.enter_subsection("Critical slip distance function");
        try
          {
            critical_slip_distance_function
              = std_cxx14::make_unique<Functions::ParsedFunction<dim>>(n_fields);
            critical_slip_distance_function->parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "FunctionParser failed to parse\n"
                      << "\t RSF L function\n"
                      << "with expression \n"
                      << "\t' " << prm.get("Function expression") << "'";
            throw;
          }
        prm.leave_subsection();

        quasi_static_strain_rate = prm.get_double("Quasi static strain rate");

        steady_state_velocity = prm.get_double("Steady state velocity for RSF");

        effective_normal_stress_on_fault = prm.get_double("Effective normal stress on fault");

        use_radiation_damping = prm.get_bool("Use radiation damping");

        cut_edot_ii = prm.get_bool("Cut edot_ii after radiation damping");

        use_always_yielding = prm.get_bool("Use always yielding");

        coordinate_system_RSF = Utilities::Coordinates::string_to_coordinate_system(prm.get("Coordinate system for RSF parameters"));

        prm.enter_subsection("Rate and state parameter a function");
        try
          {
            rate_and_state_parameter_a_function
              = std_cxx14::make_unique<Functions::ParsedFunction<dim>>(n_fields);
            rate_and_state_parameter_a_function->parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "FunctionParser failed to parse\n"
                      << "\t RSF a function\n"
                      << "with expression \n"
                      << "\t' " << prm.get("Function expression") << "'";
            throw;
          }
        prm.leave_subsection();

        prm.enter_subsection("Rate and state parameter b function");
        try
          {
            rate_and_state_parameter_b_function
              = std_cxx14::make_unique<Functions::ParsedFunction<dim>>(n_fields);
            rate_and_state_parameter_b_function->parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "FunctionParser failed to parse\n"
                      << "\t RSF b function\n"
                      << "with expression \n"
                      << "\t' " << prm.get("Function expression") << "'";
            throw;
          }
        prm.leave_subsection();

        // parameters for slip rate dependent rate and state friction

        ref_v_for_a = prm.get_double("Reference velocity for rate and state parameter a");

        ref_v_for_L = prm.get_double("Reference velocity for critical slip distance L");

        slope_s_for_a = prm.get_double("Slope of log dependence for rate and state parameter a");

        slope_s_for_L = prm.get_double("Slope of log dependence for critical slip distance L");

        if (use_theta())
          {
            // if rate-and-state friction is used, this index is needed, as it will be used to always assume yielding
            // conditions inside the fault. default is so high it should never unintentionally be reached.
            // TODO: make this a bit more flexible name-wise, like let the user define which materials should be
            // considered. Or which strategy. Could also be all, or take a and b as a proxy.
            // TODO: always yielding should be done where faut has > 70 or so volume percentage. Can be circumvented
            // right now by using max composition for viscosity averaging, because always yielding is applied  within
            // calculate_isostrain_viscosities in visco_plastic.cc
            AssertThrow(this->introspection().compositional_name_exists("fault"),
                        ExcMessage("Material model with rate-and-state friction only works "
                                   "if there is a compositional field that is called fault. For this composition "
                                   "yielding is always assumed due to the rate-and-state framework."));
            fault_composition_index = this->introspection().compositional_index_for_name("fault");

            AssertThrow(this->introspection().compositional_name_exists("theta"),
                        ExcMessage("Material model with rate-and-state friction only works "
                                   "if there is a compositional field that is called theta. It is "
                                   "used to store the state variable."));
            theta_composition_index = this->introspection().compositional_index_for_name("theta");
          }
      }



      template <int dim>
      void
      FrictionOptions<dim>::create_friction_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        if (out.template get_additional_output<FrictionAdditionalOutputs<dim> >() == nullptr)
          {
            const unsigned int n_points = out.n_evaluation_points();
            out.additional_outputs.push_back(
              std_cxx14::make_unique<FrictionAdditionalOutputs<dim>> (n_points));
          }
      }



      template <int dim>
      void
      FrictionOptions<dim>::
      fill_friction_outputs(const unsigned int i,
                            const std::vector<double> &volume_fractions,
                            const MaterialModel::MaterialModelInputs<dim> &in,
                            MaterialModel::MaterialModelOutputs<dim> &out,
                            const std::vector<double> edot_ii) const
      {
        FrictionAdditionalOutputs<dim> *friction_out = out.template get_additional_output<FrictionAdditionalOutputs<dim> >();

        if (friction_out != nullptr)
          {
            if (use_theta())
              {
                friction_out->RSF_a[i] = 0;
                friction_out->RSF_b[i] = 0;
                friction_out->RSF_L[i] = 0;
                friction_out->edot_ii[i] = 0;

                for (unsigned int j=0; j < volume_fractions.size(); ++j)
                  {
                    friction_out->RSF_a[i] += volume_fractions[j] * calculate_depth_dependent_a_and_b(in.position[i], j).first;
                    friction_out->RSF_b[i] += volume_fractions[j] * calculate_depth_dependent_a_and_b(in.position[i], j).second;
                    friction_out->RSF_L[i] += volume_fractions[j] * get_critical_slip_distance(in.position[i], j);
                    friction_out->edot_ii[i] += volume_fractions[j] * edot_ii[j];
                  }
              }
          }
      }
    }
  }
}



// Explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
  template class FrictionAdditionalOutputs<dim>; \
  \
  namespace Rheology \
  { \
    template class FrictionOptions<dim>; \
  }

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
