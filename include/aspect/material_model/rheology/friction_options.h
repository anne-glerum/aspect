/*
  Copyright (C) 2019 by the authors of the ASPECT code.

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

#ifndef _aspect_material_model_rheology_friction_options_h
#define _aspect_material_model_rheology_friction_options_h

#include <aspect/global.h>
#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/base/parsed_function.h>
#include <aspect/utilities.h>

#include <aspect/material_model/rheology/drucker_prager.h>
#include <deal.II/fe/component_mask.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * Additional output fields for the rate-and-state parameters to be added
     * to the MaterialModel::MaterialModelOutputs structure and filled in the
     * MaterialModel::Interface::evaluate() function.
     */
    template <int dim>
    class FrictionAdditionalOutputs : public NamedAdditionalMaterialOutputs<dim>
    {
      public:
        explicit FrictionAdditionalOutputs(const unsigned int n_points);

        std::vector<double> get_nth_output(const unsigned int idx) const override;

        /**
         * The value of the rate and state friction parameter a.
         */
        std::vector<double> RSF_a;

        /**
         * The value of the rate and state friction parameter b.
         */
        std::vector<double> RSF_b;

        /**
         * The value of the rate and state friction parameter L, the critical
         * slip distance.
         */
        std::vector<double> RSF_L;

        /**
         * the current edot ii - second invariant of the deviatoric stress tensor
         */
        std::vector<double> edot_ii;
    };



    namespace Rheology
    {
      /**
       * Enumeration for selecting which type of friction dependence to use.
       *
       * For the type 'independent', the user-supplied internal angle of friction is used.
       *
       * For the type 'dynamic friction', the friction angle is rate dependent following
       * Equation 13 from \\cite{van_dinther_seismic_2013}.
       *
       * For the type 'rate and state dependent friction',  the friction angle is calculated
       * using classic aging rate-and-state friction by Ruina (1983) as described in
       * Equations (4--7) in \\cite{sobolev_modeling_2017}.
       *
       * WIP: For the type 'rate and state dependent friction plus linear slip weakening',
       * the rate-and-state dependent friction is expanded by a linear slip weakening
       * copmonent. This is done following Appendix A in \\cite{sobolev_modeling_2017}.
       * Therein linear slip weakening is added 'as half or more of the friction weakening
       * may be in fact related to other than RSF type of friction-weakening mechanisms'.
       *
       * For the type 'slip rate dependent rate and state dependent friction', the rate
       * and state parameter 'a' and the critical slip distance L are slip rate dependent
       * following Equations 8 and 9 in \\cite{im_slip-rate-dependent_2020}. Slip rate
       * dependent friction parameters seem to produce better results for the occurrence
       * conditions of slow slip events.
       *
       * For the type 'regularized rate and state dependent friction', the friction angle
       * is computed following the high velocity approximation of the classic rate and
       * state friction as in \\cite{herrendorfer_invariant_2018}. This overcomes the
       * problem of ill-posedness and the possibility of negative friction for very small
       * velocities.
       *
       * For the type 'steady state rate and state dependent friction', the friction angle
       * is computed as the steady-state friction coefficient in rate-and-state friction
       * that is reached when state evolves toward a steady state \\theta_{ss} = L/V
       * at constant slip velocities.
       *
       * Strain-weakening and friction dependence mechanisms other than rate or state
       * dependence are handled outside this functionality.
       * TODO: an option to use dynamic friction and rate-and-state friction together
       */
      enum FrictionDependenceMechanism
      {
        independent,
        dynamic_friction,
        rate_and_state_dependent_friction,
        rate_and_state_dependent_friction_plus_linear_slip_weakening,
        slip_rate_dependent_rate_and_state_dependent_friction,
        regularized_rate_and_state_dependent_friction,
        steady_state_rate_and_state_dependent_friction
      };

      template <int dim>
      class FrictionOptions : public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Declare the parameters this function takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm);

          /**
           * A function that computes the new friction angle when rate and/or state
           * dependence is taken into account. Given a volume fraction with
           * the index j and a vector of all compositional fields, it returns
           * the newly calculated friction angle.
           */
          double
          compute_dependent_friction_angle(const double current_edot_ii,
                                           const unsigned int j,
                                           const std::vector<double> &composition,
                                           typename DoFHandler<dim>::active_cell_iterator current_cell,
                                           double current_friction,
                                           const Point<dim> &position) const;

          /**
           * A function that computes the current value for the state variable theta.
           */
          double compute_theta(double theta_old,
                               const double current_edot_ii,
                               const double cellsize,
                               const double critical_slip_distance,
                               const Point<dim> &position) const;

          /**
           * A function that fills the reaction terms for the state variable theta in
           * the MaterialModelOutputs object that is handed over if inside RSF material.
           */
          void compute_theta_reaction_terms(const int q,
                                            const std::vector<double> &volume_fractions,
                                            const MaterialModel::MaterialModelInputs<dim> &in,
                                            const double min_strain_rate,
                                            const double ref_strain_rate,
                                            bool use_elasticity,
                                            bool use_reference_strainrate,
                                            const double &elastic_shear_moduli,
                                            const double dte,
                                            MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * A function that returns the selected type of friction dependence.
           */
          FrictionDependenceMechanism
          get_friction_dependence_mechanism () const;

          /**
           * A function that returns if the state variable theta is used.
           */
          bool use_theta() const;

          /**
           * Function to calculate depth-dependent a and b values for state-dependent friction
           * at a certain depth and for composition j.
           */
          std::pair<double,double>
          calculate_depth_dependent_a_and_b(const Point<dim> &position, const int j) const;

          /**
           * Function that gets the critical slip distance at a certain position for composition j.
           */
          double get_critical_slip_distance(const Point<dim> &position, const int j) const;

          /**
           * Function that gets the effective friction factor at a certain position for composition j.
           */
          double get_effective_friction_factor(const Point<dim> &position) const;

          /**
           * Create the additional material model outputs object that contains the
           * rate-and-state friction parameters.
           */
          void create_friction_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * A function that fills the friction parameters additional output in the
           * MaterialModelOutputs object that is handed over, if it exists.
           * Does nothing otherwise.
           */
          void fill_friction_outputs (const unsigned int point_index,
                                      const std::vector<double> &volume_fractions,
                                      const MaterialModel::MaterialModelInputs<dim> &in,
                                      MaterialModel::MaterialModelOutputs<dim> &out,
                                      const std::vector<double> edot_ii) const;

          /**
           * A value for the effective normal stress on the fault that is used in Tresca friction
           * which is available under the yield mechanism tresca.
           */
          double effective_normal_stress_on_fault;

          /**
           * The field index of the compositional field "fault", which is needed if the fault material
           * is assumed to be always yielding as is the case for classical rate-and-state friciton.
           */
          unsigned int fault_composition_index;

          /**
           * The field index of the compositional field "theta".
           */
          unsigned int theta_composition_index;

          /**
           * Whether to include radiation damping as a representation of energy outflow due to
           * seismic waves.
           */
          bool use_radiation_damping;

          /**
           *  Whether to print negative/Zero (old) thetas. ToDo: This should only be a temporary feature!
           */
          bool print_thetas;

          /**
           * Whether or not to cut edot ii after radiation damping.
           */
          bool cut_edot_ii;

          /**
           * Whether to assume always yielding in the fault material.
           */
          bool use_always_yielding;

        private:
          /**
           * Input parameters for the drucker prager plasticity.
           */
          Rheology::DruckerPragerParameters drucker_prager_parameters;

          /*
           * Object for computing plastic stresses, viscosities, and additional outputs
           */
          Rheology::DruckerPrager<dim> drucker_prager_plasticity;

          /**
           * Select the mechanism to be used for the friction dependence.
           * Possible options: none | dynamic friction | rate and state dependent friction
           * | rate and state dependent friction plus linear slip weakening | slip rate
           * dependent rate and state dependent friction | regularized rate and state
           * dependent friction
           */
          FrictionDependenceMechanism friction_dependence_mechanism;

          /**
           * Dynamic friction input parameters
           */

          /**
           * Dynamic angles of internal friction that are used at high strain rates.
           */
          std::vector<double> dynamic_angles_of_internal_friction;

          /**
           * The characteristic strain rate value at which the angle of friction is taken as
           * the mean of the dynamic and the static angle of friction. When the effective
           * strain rate in a cell is very high the dynamic angle of friction is taken, when
           * it is very low the static angle of internal friction is chosen.
           */
          double dynamic_characteristic_strain_rate;

          /**
           * An exponential factor in the equation for the calculation of the friction angle
           * to make the transition between static and dynamic friction angle more smooth or
           * more step-like.
           */
          double dynamic_friction_smoothness_exponent;

          /**
           * Rate-and-state friction input parameters
           */

          /**
          * A number that is multiplied with the coefficient of friction to take into
           * account the influence of pore fluid pressure. This makes the friction
           * coefficient an effective friction coefficient as in Sobolev and Muldashev (2017).
           */
          Functions::ParsedFunction<dim> effective_friction_factor_function;

          /**
           * The critical slip distance in rate-and-state friction. Used to calculate the state
           * variable theta.
           */
          std::unique_ptr<Functions::ParsedFunction<dim> > critical_slip_distance_function;

          /**
           * Arbitrary slip rate at which friction equals the reference friction angle in
           * rate-and-state friction. There are different names for this parameter in the
           * literature varying between steady state slip rate, reference slip rate,
           * quasi-static slip rate.
           */
          double RSF_ref_velocity;

          /**
           * The velocity used in 'steady state rate and state dependent friction' that is
           * assumed to have remained constant over a long period such that the friction angle
           * evolved to a steady-state.
           */
          double steady_state_velocity;

          /**
           * Parsed functions that specify the rate-and-state parameters a and b which must be
           * given in the input file using the function method.
           */
          std::unique_ptr<Functions::ParsedFunction<dim> > rate_and_state_parameter_a_function;
          std::unique_ptr<Functions::ParsedFunction<dim> > rate_and_state_parameter_b_function;

          /**
           * The coordinate representation to evaluate the functions for a, b, and critical slip
           * distance L. Possible choices are depth, cartesian and spherical.
           */
          Utilities::Coordinates::CoordinateSystem coordinate_system_RSF;

          /**
           * Slip-rate dependent rate-and-state friction
           */

          /**
           * Reference velocity for slip-rate dependence of rate-and-state parameter a.
           */
          double ref_v_for_a;

          /**
           * Reference velocity for slip-rate dependence of the critical slip distance.
           */
          double ref_v_for_L;

          /**
           * Slope for the log linear slip-rate dependence of rate-and-state parameter a.
           */
          double slope_s_for_a;

          /**
           * Slope for the log linear slip-rate dependence of the critical slip distance.
           */
          double slope_s_for_L;
      };
    }
  }
}
#endif
