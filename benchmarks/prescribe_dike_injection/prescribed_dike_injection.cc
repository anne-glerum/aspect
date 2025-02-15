/*
  Copyright (C) 2021 - 2024 by the authors of the ASPECT code.
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

#include <algorithm>
#include <vector>
#include <random>

#include "prescribed_dike_injection.h"
#include <aspect/geometry_model/box.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/utilities.h>
#include <aspect/parameters.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void
    PrescribedDikeInjection<dim>::initialize()
    {
      base_model->initialize();
      // TODO: Applies to all geometry models.
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model()),
                  ExcMessage("Currently, this function only works with the box geometry model."));
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::update()
    {
      base_model->update();

      // we get time passed as seconds (always) but may want
      // to reinterpret it in years
      if (this->convert_output_to_years())
        dike_injection_rate_function.set_time (this->get_time() / year_in_seconds);
      else
        dike_injection_rate_function.set_time (this->get_time());

      // If using random dike generation
      if (enable_random_dike_generation)
        {
          // Dike is randomly generated in the potential dike generation
          // zone at each timestep.
          double x_location_random_dike = 0.0;
          double top_depth_change_random_dike = 0.0;

          // 1. generate a random number
          // We prescribe a seed for the random generator to guarantee the same
          // random number on each mpi process and when a simulation is return.
          std::mt19937 random_number_generator (static_cast<unsigned int>((seed + 1) * this->get_timestep_number()));
          std::uniform_real_distribution<> dist(0, 1.0);

          // 2.1 Randomly generate the dike location (x_coordinate) by applying
          // quadratic transfer function, which is a parabolic relationship
          // between the random number and the dike x-coordinate.
          // i.e., random_number = (coefficent_a *
          //                        (random_dike_x_location_raw - 
          //                         (x_center_dike_generation_zone - width_dike_generation_zone / 2)))^2
          // coefficent_a = 1 / (width_dike_generation_zone / 2)
          double x_location_random_dike_raw = 0.5 * width_dike_generation_zone
                                              * std::sqrt(dist(random_number_generator))
                                              + x_center_dike_generation_zone 
                                              - 0.5 * width_dike_generation_zone;

          // 2.2 Randomly generate the dike top depth variation by applying
          // the same function.
          double top_depth_change_random_dike_raw = 0.5 * range_depth_change_random_dike
                                                    * std::sqrt(dist(random_number_generator))
                                                    + ini_top_depth_random_dike 
                                                    - 0.5 * range_depth_change_random_dike;

          // Randomly select the left or right part of the dike generation zone around
          // the center position of the zone (x_center_dike_generation_zone).
          std::uniform_real_distribution<> dist2(0,1.0);
          double second_random_number = dist2(random_number_generator);
          if (second_random_number < 0.5)
            {
              x_location_random_dike = x_location_random_dike_raw;
              top_depth_change_random_dike = top_depth_change_random_dike_raw;
            }
          else
            {
              x_location_random_dike = 2 * x_center_dike_generation_zone - x_location_random_dike_raw;
              top_depth_change_random_dike = 2 * ini_top_depth_random_dike - top_depth_change_random_dike_raw;
            }

          // 3. Find the x-direction side boundaries of the column where the dike is located.
          // Currently, we only consider the box geometry.
          const GeometryModel::Box<dim> &
          geometry_model
            = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());

          // Get the maximum resolution in the x direction.
          const double dx_max = geometry_model.get_extents()[0] 
                                / (geometry_model.get_repetitions()[0]
                                * std::pow(2,total_refinement_levels));

          // Currently, we recommend that the dike's width should ideally be 
          // equal to the highest x-resolution or an integer multiple thereof.
          x_left_boundary_random_dike = std::floor(x_location_random_dike / dx_max) * dx_max;
          x_right_boundary_random_dike = x_left_boundary_random_dike + width_random_dike;
          top_depth_random_dike = ini_top_depth_random_dike + top_depth_change_random_dike;
        }
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      PrescribedPlasticDilation<dim>
      *prescribed_dilation = (this->get_parameters().enable_prescribed_dilation)
                              ? out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()
                              : nullptr;
      ReactionRateOutputs<dim>
      *reaction_rate_out = (this->get_parameters().use_operator_splitting)
                            ? out.template get_additional_output<MaterialModel::ReactionRateOutputs<dim>>()
                            : nullptr;
      // Initiallize reaction_rates to 0.0.
      if (reaction_rate_out != nullptr)
        for (auto &row : reaction_rate_out->reaction_rates)
          std::fill(row.begin(), row.end(), 0.0);

      // When calculating other properties such as viscosity, we need to 
      // correct for the effect of injection on the strain rate and thus
      // the deviatoric strain rate.
      MaterialModel::MaterialModelInputs<dim> in_corrected_strainrate (in);

      // Store dike injection rate for each evaluation point.
      std::vector<double> dike_injection_rate(in.n_evaluation_points(), 0.);

      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          // First, we find the location of the dike which is either randomly
          // generated or prescribed by the user, then give the dike_injection_rate
          // to evaluation points in the dike area.
          if (enable_random_dike_generation)
            {
              // Note: when the dike is generated randomly, the prescribed
              // injection rate in the 'Dike injection rate function' should be
              // time-dependent only and independent of spatial coordinates.
              const double point_depth = this->get_geometry_model().depth(in.position[i]);

              // Determine whether the current cell lies within the dike boundaries.
              // Note the injection starts from Timestep 1.
              if (in.position[i][0] >= x_left_boundary_random_dike
                  && in.position[i][0] <= x_right_boundary_random_dike
                  && in.temperature[i] <= T_bottom_dike 
                  && point_depth >= std::max(top_depth_random_dike, 0.0)
                  && this->get_timestep_number() > 0)
                dike_injection_rate[i] = this->convert_output_to_years()
                                         ? dike_injection_rate_function.value(in.position[i]) / year_in_seconds
                                         : dike_injection_rate_function.value(in.position[i]);

              // Dike injection effect removal
              // Ensure the current cell is located within the dike area.
              if (dike_injection_rate[i] > 0.0
                  && this->get_timestep_number() > 0
                  && in.current_cell.state() == IteratorState::valid
                  && in.current_cell->center()[0] >= x_left_boundary_random_dike
                  && in.current_cell->center()[0] <= x_right_boundary_random_dike)
                in_corrected_strainrate.strain_rate[i][0][0] -= dike_injection_rate[i];
            }
          else
            {
              // User-defined dikes. 
              // The 'Dike injection rate function' is related to both the time and
              // spatial coordinates. Currently, we assume that the bottom depth
              // of the dike is limited by the user-set isothermal depth of the
              // brittle-ductile transition (BDT).
              if (in.temperature[i] <= T_bottom_dike
                  && this->get_timestep_number() > 0)
                dike_injection_rate[i] = this->convert_output_to_years()
                                         ? dike_injection_rate_function.value(in.position[i]) / year_in_seconds
                                         : dike_injection_rate_function.value(in.position[i]);

              // Dike injection effect removal
              if (dike_injection_rate[i] > 0.0
                  && in.current_cell.state() == IteratorState::valid
                  && dike_injection_rate_function.value(in.current_cell->center()) > 0.0)
                in_corrected_strainrate.strain_rate[i][0][0] -= dike_injection_rate[i];
            }
        }
      
      // Fill variable out with the results from the base material model
      // using the corrected strain rate model input.
      base_model->evaluate(in_corrected_strainrate, out);

      // Below we start to track the motion of the dike injection material.
      AssertThrow(this->introspection().compositional_name_exists("injection_phase"),
                  ExcMessage("Material model 'prescribed dike injection' only works if "
                             "there is a compositional field called 'injection_phase'. "));

      // Index for injection phase
      unsigned int injection_phase_index = this->introspection().compositional_index_for_name("injection_phase");

      // Indices for all chemical compositional fields, and not e.g., plastic strain.
      const std::vector<unsigned int> chemical_composition_indices = this->introspection().chemical_composition_field_indices();

      const auto &component_indices = this->introspection().component_indices.compositional_fields;

      // Positions of quadrature points at the current cell
      std::vector<Point<dim>> quadrature_positions;

      // The newly injected material will replace part of the original material
      // for the existence duration of a dike.
      double injected_material_amount = 0.0;

      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          // Activate the dike injection by adding the additional RHS 
          // terms of injection to Stokes equations.
          if (prescribed_dilation != nullptr)
            prescribed_dilation->dilation[i] = dike_injection_rate[i];

          // The amount of newly injected material is either user-set or
          // time-dependent (default) which equals the product of the injection
          // rate and the current timestep.
          if (this->simulator_is_past_initialization())
            injected_material_amount = dike_injection_rate[i] * this->get_timestep();

          // The amount of newly injected material can be user-set.
          if (prescribed_material_injection_amount != 0.0)
            injected_material_amount = prescribed_material_injection_amount;

          if (dike_injection_rate[i] > 0.0
              && this->get_timestep_number() > 0
              && in.current_cell.state() == IteratorState::valid)
            {
              // We need to obtain the values of chemical compositional fields
              // at the previous time step, as the values from the current
              // linearization point are an extrapolation of the solution from
              // the old timesteps.
              // Prepare the field function and extract the old solution values at the current cell.
              std::vector<Point<dim>> quadrature_positions(1,this->get_mapping().transform_real_to_unit_cell(in.current_cell, in.position[i]));

               // Use a small_vector to avoid memory allocation if possible.
              small_vector<double> old_solution_values(this->get_fe().dofs_per_cell);
              in.current_cell->get_dof_values(this->get_old_solution(),
                                              old_solution_values.begin(),
                                              old_solution_values.end());

              // If we have not been here before, create one evaluator for each compositional field
              if (composition_evaluators.size() == 0)
                composition_evaluators.resize(this->n_compositional_fields());

              // Make sure the evaluators have been initialized correctly, and have not been tampered with
              Assert(composition_evaluators.size() == this->n_compositional_fields(),
                    ExcMessage("The number of composition evaluators should be equal to the number of compositional fields."));

              // Loop only over the chemical compositional fields
              for (unsigned int c : chemical_composition_indices)
                {
                  // Only create the evaluator the first time we get here
                  if (!composition_evaluators[c])
                    composition_evaluators[c]
                      = std::make_unique<FEPointEvaluation<1, dim>>(this->get_mapping(),
                                                                    this->get_fe(),
                                                                    update_values,
                                                                    component_indices[c]);

                  composition_evaluators[c]->reinit(in.current_cell, quadrature_positions);
                  composition_evaluators[c]->evaluate({old_solution_values.data(),old_solution_values.size()},
                                                      EvaluationFlags::values);
                  double old_solution_composition = composition_evaluators[c]->get_value(0);

                  if (c == injection_phase_index)
                    {
                      // If the value increases to greater than 1, no longer let it increase.
                      if (old_solution_composition + injected_material_amount >= 1.0)
                        out.reaction_terms[i][c] = 0.0;
                      else
                        out.reaction_terms[i][c] = injected_material_amount;

                      // Fill reaction rate outputs instead of the reaction terms if
                      // we use operator splitting (and then set the latter to zero).
                      if (reaction_rate_out != nullptr)
                        {
                          reaction_rate_out->reaction_rates[i][c] = out.reaction_terms[i][c]
                                                                    / this->get_timestep();
                          out.reaction_terms[i][c] = 0.0;
                        }
                    }
                  else
                    {
                      // When newly dike material is injected, the other compositional fields
                      // at the dike point will be reduced in the same proportion (p_c) to
                      // ensure that the sum of all compositional fields is always 1.0.
                      // For example, in the previous step, the dike material has a compositional
                      // field with the value of c_dike_old and another compositional field
                      // with a value of c_1_old. So, c_dike_old + c_1_old = 1.0. Here we
                      // leave the background field alone, because it will be automatically
                      // populated if c_dike_old + c_1_old < 1.0.
                      // In the current step, when adding new dike material of amount 'c_dike_add',
                      // c_dike_new =  c_dike_old + c_dike_add. c_1_new = c_1_old * p_c.
                      // Since c_1_new + c_dike_new = 1.0, we get 
                      // p_c = (1.0 - c_dike_old - c_dike_add) / c_1_old.
                      // Then the amount of change in c_1 is:
                      // delta_c_1 = c_1_new - c_1_old = c_1_old * (p_c - 1.0)
                      // = - c_1_old * (c_dike_add / (1.0001 - c_dike_old))
                      // To avoid dividing by 0, we will use 1.0001 instead of 1.0.

                      // Only create the evaluator the first time if we have not been to injection_phase composition before.
                      if (!composition_evaluators[injection_phase_index])
                        composition_evaluators[injection_phase_index]
                          = std::make_unique<FEPointEvaluation<1, dim>>(this->get_mapping(),
                                                                        this->get_fe(),
                                                                        update_values,
                                                                        component_indices[injection_phase_index]);

                      composition_evaluators[injection_phase_index]->reinit(in.current_cell, quadrature_positions);
                      composition_evaluators[injection_phase_index]->evaluate({old_solution_values.data(),old_solution_values.size()},
                                                                               EvaluationFlags::values);

                      // We limit the value of the injection phase compositional field from the previous timestep to [0,1].
                      double old_injection_phase_composition = std::max(std::min(composition_evaluators[injection_phase_index]->get_value(0),1.0),0.0);

                      out.reaction_terms[i][c] = -old_solution_composition 
                                                 * std::min(injected_material_amount / (1.0001 - old_injection_phase_composition), 1.0);

                      // Fill reaction rate outputs instead of the reaction terms if
                      // we use operator splitting (and then set the latter to zero).
                      if (reaction_rate_out != nullptr)
                        {
                          reaction_rate_out->reaction_rates[i][c] = out.reaction_terms[i][c]
                                                                    / this->get_timestep();
                          out.reaction_terms[i][c] = 0.0;
                        }
                    }
                }

              // If the "single Advection" nonlinear solver scheme is used, 
              // it is necessary to set the strain reaction terms to 0 to avoid
              // additional plastic deformation generated by dike injection
              // within the dike zone.
              if (this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_single_Stokes
                  ||
                  this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_iterated_Stokes
                  ||
                  this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes
                  ||
                  this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_iterated_defect_correction_Stokes)
                {
                  if (this->introspection().compositional_name_exists("plastic_strain"))
                    out.reaction_terms[i][this->introspection().compositional_index_for_name("plastic_strain")] = 0.0;
                  if (this->introspection().compositional_name_exists("viscous_strain"))
                    out.reaction_terms[i][this->introspection().compositional_index_for_name("viscous_strain")] = 0.0;
                  if (this->introspection().compositional_name_exists("total_strain"))
                    out.reaction_terms[i][this->introspection().compositional_index_for_name("total_strain")] = 0.0;
                  if (this->introspection().compositional_name_exists("noninitial_plastic_strain"))
                    out.reaction_terms[i][this->introspection().compositional_index_for_name("noninitial_plastic_strain")] = 0.0;
                  //TODO: Check and test elastic stress reaction terms.
                }
            }

        }
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Prescribed dike injection");
        {
          prm.declare_entry("Base model","simple",
                            Patterns::Selection(MaterialModel::get_valid_model_names_pattern<dim>()),
                            "The name of a material model that will be modified by an "
                            "averaging operation. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Material models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.declare_entry("Dike material injection amount", "0.0", Patterns::Double(0),
                            "Prescribed the amount of newly injected material from the dike. "
                            "Units: none.");
          prm.declare_entry("Dike bottom temperature", "873.0", Patterns::Double(0),
                            "Temperature that defines the bottom of the generated dikes. It is "
                            "usually set to the temperature of the brittle-ductile transition (BDT)"
                            "zone. Units: K.");
          prm.declare_entry("X center of the dike generation zone", "0.0", Patterns::Double(0),
                            "X_coordinate of the center of the dike generation zone. Units: m.");
          prm.declare_entry("Width of the dike generation zone", "0.0", Patterns::Double(0),
                            "Width of the dike generation zone. Units: m.");
          prm.declare_entry("Random number generator seed", "0", Patterns::Double(0),
                            "The value of the seed used for the random number generator. Units: none.");
          prm.declare_entry("Initial top depth of the randomly generated dike", "0.0", Patterns::Double(0),
                            "Initial top depth of the randomly generated dike. Units: m.");
          prm.declare_entry("Range of depth variation in randomly generated dikes", "0.0", Patterns::Double(0),
                            "Range of depth variation in randomly generated dikes. Units: m.");
          prm.declare_entry("Width of the randomly generated dike", "0.0", Patterns::Double(0),
                            "Width of the generated dike. Currently, we recommend that the dike's width should "
                            "ideally be equal to the highest x-resolution or an integer multiple thereof. "
                            "Units: m.");
          prm.declare_entry("Enable random dike generation", "false", Patterns::Bool (),
                            "Whether the dikes are generated randomly. If the dike is generated randomly, "
                            "the prescribed injection rate in the 'Dike injection rate function' should be "
                            "only time dependent and independent of the xyz-coordinate.");

          // Note: when the dike is generated randomly, the prescribed
          // injection rate in the 'Dike injection rate function' should be
          // time-dependent only and independent of spatial coordinates.
          prm.enter_subsection("Dike injection rate function");
          {
            Functions::ParsedFunction<dim>::declare_parameters(prm,1);
            prm.declare_entry("Function expression","0.0");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Prescribed dike injection");
        {
          AssertThrow( prm.get("Base model") != "Prescribed dike injection",
                       ExcMessage("You may not use ''prescribed dike injection'' as the base model for itself."));

          // create the base model and initialize its SimulatorAccess base
          // class; it will get a chance to read its parameters below after we
          // leave the current section
          base_model = create_material_model<dim>(prm.get("Base model"));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          prescribed_material_injection_amount = prm.get_double ("Dike material injection amount");
          T_bottom_dike = prm.get_double ("Dike bottom temperature");
          enable_random_dike_generation = prm.get_bool("Enable random dike generation");
          x_center_dike_generation_zone = prm.get_double ("X center of the dike generation zone");
          width_dike_generation_zone = prm.get_double ("Width of the dike generation zone");
          seed = prm.get_double ("Random number generator seed");
          ini_top_depth_random_dike = prm.get_double ("Initial top depth of the randomly generated dike");
          range_depth_change_random_dike = prm.get_double ("Range of depth variation in randomly generated dikes");
          width_random_dike = prm.get_double ("Width of the randomly generated dike");

          prm.enter_subsection("Dike injection rate function");
          {
            try
              {
                dike_injection_rate_function.parse_parameters(prm);
              }
            catch (...)
              {
                std::cerr << "ERROR: FunctionParser failed to parse\n"
                          << "\t Dike injection rate function\n"
                          << "with expression \n"
                          << "\t' " << prm.get("Function expression") << "'";
                throw;
              } 
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // The total refinment levels in the model.
      prm.enter_subsection("Mesh refinement");
      {
        total_refinement_levels = prm.get_integer("Initial adaptive refinement") + prm.get_integer("Initial global refinement");
      }
      prm.leave_subsection();

      // If 'Free Surface' is used, please ensure the 'Surface velocity
      // projection' is vertical. If the projection is normal, which means
      // the surface mesh can deform both horizontally and vertically, this
      // may distort a surface element.
      prm.enter_subsection("Mesh deformation");
      {
        // Check if "free surface" is specified in Mesh deformation boundary indicators
        std::string boundary_indicators = prm.get("Mesh deformation boundary indicators");
        std::string advection_direction = "nan"; 
        if (boundary_indicators.find("free surface") != std::string::npos)
        {
            prm.enter_subsection("Free surface");
            {
              advection_direction = prm.get("Surface velocity projection");
              AssertThrow(advection_direction == "vertical",
                          ExcMessage("The projection is " + advection_direction + 
                                    ". However, this function currently prefers to use "
                                    "vertical projection if using free surface."));              
            }
            prm.leave_subsection();
        }
      }
      prm.leave_subsection(); 

      // Parse parameters related to the base model.
      base_model->parse_parameters(prm);
      this->model_dependence = base_model->get_model_dependence();
    }

    template <int dim>
    bool
    PrescribedDikeInjection<dim>::
    is_compressible () const
    {
      return base_model->is_compressible();
    }

    template <int dim>
    void
    PrescribedDikeInjection<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // The base model may have additional outputs, so we need to copy
      // these additional outputs.
      base_model->create_additional_named_outputs(out);

      //Stokes additional RHS for prescribed dilation
      const unsigned int n_points = out.n_evaluation_points();
      if (this->get_parameters().enable_prescribed_dilation
          && out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedPlasticDilation<dim>> (n_points));
        }

      AssertThrow(!this->get_parameters().enable_prescribed_dilation
                  ||
                  out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim> >()->dilation.size()
                  == n_points, ExcInternalError());

      if (this->get_parameters().use_operator_splitting
          && out.template get_additional_output<MaterialModel::ReactionRateOutputs<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::ReactionRateOutputs<dim>> (n_points,
                                                                        this->n_compositional_fields()));
        }
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PrescribedDikeInjection,
                                   "prescribed dike injection",
                                   "The material model uses a ``Base model'' from which "
                                   "material properties are derived. It adds source "
                                   "terms in the Stokes equations to implement the dike "
                                   "intrusion process in the model. It also adds the "
                                   "material for a dike field.")
  }
}