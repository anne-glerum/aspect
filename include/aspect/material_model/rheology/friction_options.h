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


#include <aspect/material_model/rheology/strain_dependent.h>
#include <aspect/material_model/rheology/drucker_prager.h>
#include <aspect/material_model/rheology/elasticity.h>

#include<deal.II/fe/component_mask.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    namespace Rheology
    {
      /**
       * Enumeration for selecting which type of friction dependence to use.
       * For independent, internal angle of friction is used.
       * Otherwise, the friction angle can be rate and/or state dependent.
       */

      enum FrictionDependenceMechanism
      {
        independent,
        dynamic_friction,
        state_dependent_friction
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
                                           double current_friction) const;

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
                                            const std::vector<double> &volume_fractions,
                                            const double min_strain_rate,
                                            const double ref_strain_rate,
                                            bool use_elasticity,
                                            bool use_reference_strainrate,
                                            const std::vector<double> &elastic_shear_moduli,
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
          bool get_theta_in_use() const;

        private:

          FrictionDependenceMechanism friction_dependence_mechanism;


          /*
           * Objects for computing plastic stresses, viscosities, and additional outputs
           */
          Rheology::DruckerPrager<dim> drucker_prager_plasticity;
          /**
          * Input parameters for the drucker prager plasticity.
          */
          Rheology::DruckerPragerParameters drucker_prager_parameters;

          /**
          * Object for computing viscoelastic viscosities and stresses.
          */
          Rheology::Elasticity<dim> elastic_rheology;

          Rheology::StrainDependent<dim> strain_rheology;

          /**
          * dynamic friction input parameters
          */
          std::vector<double> dynamic_angles_of_internal_friction;
          double dynamic_characteristic_strain_rate;
          double dynamic_friction_smoothness_exponent;

          /**
          * rate and state friction input parameters
          */
          std::vector<double> rate_and_state_parameter_a;
          std::vector<double> rate_and_state_parameter_b;
          std::vector<double> effective_friction_factor;
          double critical_slip_distance;
          std::vector<double> steady_state_strain_rate;
      };
    }
  }
}
#endif


