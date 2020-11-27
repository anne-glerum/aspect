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
#include <aspect/material_model/rheology/elasticity.h>

#include <deal.II/fe/component_mask.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    namespace Rheology
    {
      /**
       * Enumeration for selecting which type of friction dependence to use.
       * For the type 'independent', the user-supplied internal angle of friction is used.
       * For the type 'dynamic friction' the friction angle is rate dependent using
       * Equation 13 from van Dinther et al. (2013).
       * For the type 'rate and state dependent friction'  the friction angle is calculated
       * using classic aging rate-and-state friction by Ruina (1983) as described in
       * Equations (4--7) in Sobolev and Muldashev (2017).
       * Strain-weakening and friction dependence mechanisms other than rate or state
       * dependence are handled outside this functionality.
       * TODO: an option to use dynamic friction and rate and state friction together
       */
      enum FrictionDependenceMechanism
      {
        independent,
        dynamic_friction,
        rate_and_state_dependent_friction
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
           * dependence is taken into account. Given a compositional field with
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
           * A function that returns a ComponentMask, which indicates that the component
           * associated with theta should be excluded during the volume fraction computation.
           */
          ComponentMask get_theta_composition_mask(ComponentMask composition_mask) const;


          /**
           * A function that computes the current value for the state variable theta.
           */
          double compute_theta(const double theta_old,
                               const double current_edot_ii,
                               const double cellsize) const;


          /**
           * A function that fills the reaction terms for the state variable theta in
           * MaterialModelOutputs object that is handed over.
           */
          void compute_theta_reaction_terms(const int q,
                                            const MaterialModel::MaterialModelInputs<dim> &in,
                                            const double min_strain_rate,
                                            const double ref_strain_rate,
                                            bool use_elasticity,
                                            bool use_reference_strainrate,
                                            const double &elastic_shear_moduli,
                                            MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * A function that returns the selected type of friction dependence.
           */
          FrictionDependenceMechanism
          get_friction_dependence_mechanism () const;

          /**
           * A function that returns if the state variable theta is used.
           */
          bool get_use_theta() const;

          /**
           * A function that returns if radiation damping is used.
           */
          bool get_use_radiation_damping() const;

        private:
          /*
           * Objects for computing plastic stresses, viscosities, and additional outputs
           */
          Rheology::DruckerPrager<dim> drucker_prager_plasticity;

          /**
          * Input parameters for the drucker prager plasticity.
          */
          Rheology::DruckerPragerParameters drucker_prager_parameters;

          /**
          * Object for computing elasticity parameters.
          */
          Rheology::Elasticity<dim> elastic_rheology;

          /**
           * Function to calculate depth-dependent a and b values for state dependent friction.
           */
          std::pair<double,double>
          calculate_depth_dependent_a_and_b(const Point<dim> &position, const int j) const;

          /**
           * Parameter about what mechanism should be used for the friction dependence.
           * Possible options: (rate-and-state) independent | dynamic friction | rate and state dependent
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
           * The characteristic strain rate value, where the angle of friction takes the mean
           * of the dynamic and the static angle of friction. When the effective strain rate
           * in a cell is very high the dynamic angle of friction is taken, when it is very low
           * the static angle of internal friction is chosen.
           */
          double dynamic_characteristic_strain_rate;

          /**
           * An exponential factor in the equation for the calculation of the friction angle
           * to make the transition between static and dynamic friction angle more smooth or
           * more step-like.
           */
          double dynamic_friction_smoothness_exponent;

          /**
          * rate and state friction input parameters
          */

          /**
           * A number that is multiplied with the coefficient of friction to take into
           * account the influence of pore fluid pressure. This makes the friction
           * coefficient an effective friction coefficient as in Sobolev and Muldashev (2017).
           */
          std::vector<double> effective_friction_factor;

          /**
           * The critical slip distance in rate and state friction. Used to calculate the state
           * variable theta.
           */
          double critical_slip_distance;

          /**
           * Arbitrary strain rate at which friction equals the reference friction angle in
           * rate and state friction.
           */
          double quasi_static_strain_rate;

          /**
           * Whether to include radiation damping as a representation of energy outflow due to
           * seismic waves.
           */
          bool use_radiation_damping;

          /**
           * Parsed functions that specify a and b depth-dependence when using the Function
           * method.
           */
          std::unique_ptr<Functions::ParsedFunction<dim> > rate_and_state_parameter_a_function;
          std::unique_ptr<Functions::ParsedFunction<dim> > rate_and_state_parameter_b_function;

          /**
           * The coordinate representation to evaluate the function. Possible
           * choices are depth, cartesian and spherical.
           */
          Utilities::Coordinates::CoordinateSystem coordinate_system_a;
          Utilities::Coordinates::CoordinateSystem coordinate_system_b;
      };
    }
  }
}
#endif


