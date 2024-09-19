/*
  Copyright (C) 2024 by the authors of the ASPECT code.
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
#ifndef _aspect_material_model_prescribed_dike_injection_h
#define _aspect_material_model_prescribed_dike_injection_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * This dike injection function defines material injection throug a narrow
     * dike by prescribing a dilation term applied to the mass equation, a
     * deviatoric strain rate correction term in the momentum equation, and
     * a injection-related heating term in the temperature equation.
     * Since the direction of the dike opening is assumed to be the same as 
     * the direction of plate spreading, there should be no volumetric 
     * deformation in the dike, i.e., dike injection has no effect on the
     * deviatoric strain rate.
     *
     * @ingroup MaterialModels
     */

    template <int dim>
    class PrescribedDikeInjection : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Initialize the model at the beginning of the run.
         */
        void initialize() override;

        /**
         * Update the base model and dilation function at the beginning
         * of each timestep.
         */
        void update() override;

        /**
         * Function to compute the material properties in @p out given
         * the inputs in @p in.
         */  
        void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                      MaterialModel::MaterialModelOutputs<dim> &out) const override;
        /**
         * Declare the parameters through input files.
         */
        static void
        declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters through the input file
         */
        void
        parse_parameters (ParameterHandler &prm) override;

        /**
         * Indicate whether material is compressible only based on
         * the base model.
         */
        bool is_compressible () const override;

        void
        create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const override;

      private:
        /**
         * Parsed function that specifies the dike material injection rate
         */
        Functions::ParsedFunction<dim> dike_injection_rate_function;

        /**
         * The amount of newly injected material from the dike
         */
        double prescribed_material_injection_amount;

        /**
         * Currently, we assume that the bottom depth of the generated dike is
         * is the isothermal depth of the brittle-ductile transition (BDT).
         * Therefore, here we set the temperature at the bottom of that dike
         * this isotherm.
         */
        double T_bottom_dike;
        
        /**
         * Whether to use the random dike generation (if true) or not (if false).
         * If true, the location and height of the dikes are varied randomly
         * within the dike generation zone.
         */
        bool enable_random_dike_generation;

        /**
         * X-coordinate of the center of the dike generation zone.
         */
        double x_center_dike_generation_zone;

        /**
         * Width of the dike generation zone.
         */
        double width_dike_generation_zone;

        /**
         * X-coordinates of left and right sides of the randomly generated dikes
         */
        double x_left_boundary_random_dike;
        double x_right_boundary_random_dike;

        /**
         * Initial top depth of the randomly generated dike.
         */
        double ini_top_depth_random_dike;

        /**
         * Range of depth variation in randomly generated dikes.
         */
        double range_depth_change_random_dike;

        /**
         * Randomly genetrated dike top depth, which changes every timestep
         */
        double top_depth_random_dike;

        /**
         * Width of the randomly generated dike.
         */
        double width_random_dike;

        /**
         * Seed for the random number generator
         */
        double seed;

        /**
         * The total refinement levels in the model, which equals to
         * the sum of global refinement levels and adpative refinement
         * levels. This is used in calcuting the dike location.
         */
        double total_refinement_levels;

        /**
         * Pointer to the material model used as the base model.
         */
        std::unique_ptr<MaterialModel::Interface<dim> > base_model;

          /**
           * We cache the evaluators that are necessary to evaluate
           * compositions. By caching the evaluator, we can avoid
           * recreating them every time we need it.
           */
          mutable std::vector<std::unique_ptr<FEPointEvaluation<1, dim>>> composition_evaluators;
    };
  }
}

#endif