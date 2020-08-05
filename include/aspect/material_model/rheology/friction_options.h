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
       * For none, internal angle of friction is used.
       * Otherwise, the friction angle can be rate and/or state dependent.
       */

       enum FrictionDependenceMechanism
       {
         none,
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
          compute_dependent_friction_angle(const unsigned int j,
                                           const std::vector<double> &composition,
                                           const MaterialModel::MaterialModelInputs<dim> &in,
	                    const double ref_strain_rate,
						bool use_elasticity,
                                           const double min_strain_rate) const;

          /**
           * A function that returns a ComponentMask, which indicates that the component
           * associated with theta should be excluded during the volume fraction computation.
           */
          ComponentMask get_volumetric_composition_mask() const;

          /**
           * A function that returns current_edot_ii, which is the current second invariant
          * of the strain rate tensor.
           */
          double compute_edot_ii (const MaterialModel::MaterialModelInputs<dim> &in,
	                    const double ref_strain_rate,
						bool use_elasticity,
                                const double min_strain_rate) const;

          /**
           * A function that fills the reaction terms for the state variable theta in
           * MaterialModelOutputs object that is handed over.
           */
          void compute_theta_reaction_terms(const MaterialModel::MaterialModelInputs<dim> &in,
                                           const double min_strain_rate,
	                    const double ref_strain_rate,
						bool use_elasticity,
                                            MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * A function that returns the selected type of friction dependence.
           */
          FrictionDependenceMechanism
          get_dependence_mechanism () const;
		  
        private:


          /**
          * dynamic friction input parameters
          */
          std::vector<double> dynamic_angles_of_internal_friction;
          std::vector<double> dynamic_characteristic_strain_rate;
          std::vector<double> dynamic_friction_smoothness_exponent;

          /**
          * rate and state friction input parameters
          */
          std::vector<double> rate_and_state_parameter_a;
          std::vector<double> rate_and_state_parameter_b;
          std::vector<double> critical_slip_distance;
          std::vector<double> steady_state_strain_rate;

          // IS THERE MORE I NEED HERE OR IN PUBLIC? 
      };
    }
  }
}
#endif


