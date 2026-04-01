/*
  Copyright (C) 2015 - 2024 by the authors of the ASPECT code.

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

#include <aspect/particle/property/elastic_stress.h>
#include <aspect/material_model/visco_plastic.h>
#include <aspect/material_model/viscoelastic.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/particle/world.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      ElasticStress<dim>::ElasticStress ()
        :
        material_inputs(1,0),
        material_outputs(1,0),
        material_inputs_cell(1,0),
        material_outputs_cell(1,0)
      {}



      template <int dim>
      void
      ElasticStress<dim>::initialize ()
      {
        AssertThrow((Plugins::plugin_type_matches<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model())
                     ||
                     Plugins::plugin_type_matches<const MaterialModel::Viscoelastic<dim>>(this->get_material_model())),
                    ExcMessage("This particle property only makes sense in combination with the viscoelastic or visco_plastic material model."));

        AssertThrow(this->get_parameters().enable_elasticity == true,
                    ExcMessage ("This particle property should only be used if 'Enable elasticity' is set to true"));

        const auto &manager = this->get_particle_manager(this->get_particle_manager_index()).get_property_manager();
        AssertThrow(!manager.plugin_name_exists("composition"),
                    ExcMessage("The 'elastic stress' plugin cannot be used in combination with the 'composition' plugin."));

        material_inputs = MaterialModel::MaterialModelInputs<dim>(1, this->n_compositional_fields());

        material_outputs = MaterialModel::MaterialModelOutputs<dim>(1, this->n_compositional_fields());

        material_inputs.requested_properties = MaterialModel::MaterialProperties::reaction_terms | MaterialModel::MaterialProperties::reaction_rates;

        // The reaction rates are stored in additional outputs
        this->get_material_model().create_additional_named_outputs(material_outputs);

        // Get the indices of those compositions that correspond to stress tensor elements.
        stress_field_indices = this->introspection().get_indices_for_fields_of_type(CompositionalFieldDescription::stress);
        AssertThrow((stress_field_indices.size() == 2*SymmetricTensor<2,dim>::n_independent_components ||
                     stress_field_indices.size() == SymmetricTensor<2,dim>::n_independent_components),
                    ExcMessage("The number of stress tensor element fields in the 'elastic stress' plugin does not equal the number of expected components."));

        // Get the indices of all compositions that do not correspond to stress tensor elements.
        std::vector<unsigned int> all_field_indices(this->n_compositional_fields());
        std::iota (std::begin(all_field_indices), std::end(all_field_indices), 0);
        std::set_difference(all_field_indices.begin(), all_field_indices.end(),
                            stress_field_indices.begin(), stress_field_indices.end(),
                            std::inserter(non_stress_field_indices, non_stress_field_indices.begin()));
      }



      template <int dim>
      void
      ElasticStress<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                           std::vector<double> &data) const
      {
        // Give each elastic stress field its initial composition if one is prescribed.
        data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xx")));

        data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_yy")));

        if (dim == 2)
          {
            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xy")));

          }
        else if (dim == 3)
          {
            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_zz")));

            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xy")));

            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xz")));

            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_yz")));
          }

        if (stress_field_indices.size() == 2*SymmetricTensor<2,dim>::n_independent_components)
          {
            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xx_old")));

            data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_yy_old")));

            if (dim == 2)
              {
                data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xy_old")));
              }
            else if (dim == 3)
              {
                data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_zz_old")));

                data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xy_old")));

                data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xz_old")));

                data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_yz_old")));
              }
          }
      }



      template <int dim>
      void
      ElasticStress<dim>::update_particle_properties(const ParticleUpdateInputs<dim> &inputs,
                                                     typename ParticleHandler<dim>::particle_iterator_range &particles) const
      {
        const std::shared_ptr<MaterialModel::ReactionRateOutputs<dim>> reaction_rate_outputs
          = material_outputs.template get_additional_output_object<MaterialModel::ReactionRateOutputs<dim>>();

        const unsigned int n_total_stress_components = stress_field_indices.size();

        unsigned int p = 0;
        for (auto &particle: particles)
          {
            material_inputs.position[0] = particle.get_location();


            material_inputs.current_cell = inputs.current_cell;

            material_inputs.temperature[0] = inputs.solution[p][this->introspection().component_indices.temperature];

            material_inputs.pressure[0] = inputs.solution[p][this->introspection().component_indices.pressure];

            for (unsigned int d = 0; d < dim; ++d)
              material_inputs.velocity[0][d] = inputs.solution[p][this->introspection().component_indices.velocities[d]];

            // Fill the non-stress composition inputs with the solution.
            for (const unsigned int &n : non_stress_field_indices)
              material_inputs.composition[0][n] = inputs.solution[p][this->introspection().component_indices.compositional_fields[n]];
            // For the stress composition we use the ve_stress_* stored on the particles.
            for (unsigned int n = 0; n < n_total_stress_components; ++n)
              material_inputs.composition[0][stress_field_indices[n]] = particle.get_properties()[this->data_position + n];

            Tensor<2,dim> grad_u;
            for (unsigned int d=0; d<dim; ++d)
              grad_u[d] = inputs.gradients[p][d];
            material_inputs.strain_rate[0] = symmetrize (grad_u);

            this->get_material_model().evaluate (material_inputs,material_outputs);

            // Apply the stress rotation to the ve_stress_* fields (not the ve_stress_*_old fields if they exist)
            // and update the corresponding material model inputs as well.
            for (unsigned int i = 0; i < SymmetricTensor<2,dim>::n_independent_components ; ++i)
              {
                particle.get_properties()[this->data_position + i] += material_outputs.reaction_terms[0][stress_field_indices[i]];
                material_inputs.composition[0][stress_field_indices[i]] += material_outputs.reaction_terms[0][stress_field_indices[i]];
              }

            // Evaluate the material model again, this time with the rotated stresses
            this->get_material_model().evaluate (material_inputs,material_outputs);

            // Add the reaction_rates * timestep = update to the corresponding stress
            // tensor components of current and old stresses.
            for (unsigned int i = 0; i < n_total_stress_components; ++i)
              particle.get_properties()[this->data_position + i] += reaction_rate_outputs->reaction_rates[0][stress_field_indices[i]] * this->get_timestep();

            ++p;
          }
      }



      template <int dim>
      UpdateTimeFlags
      ElasticStress<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      UpdateFlags
      ElasticStress<dim>::get_update_flags (const unsigned int component) const
      {
        if (this->introspection().component_masks.velocities[component] == true)
          return update_values | update_gradients;

        return update_values;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      ElasticStress<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;

        property_information.emplace_back("ve_stress_xx",1);
        property_information.emplace_back("ve_stress_yy",1);

        if (dim == 2)
          {
            property_information.emplace_back("ve_stress_xy",1);
          }
        else if (dim == 3)
          {
            property_information.emplace_back("ve_stress_zz",1);
            property_information.emplace_back("ve_stress_xy",1);
            property_information.emplace_back("ve_stress_xz",1);
            property_information.emplace_back("ve_stress_yz",1);
          }

        if  (stress_field_indices.size() == 2*SymmetricTensor<2,dim>::n_independent_components)
          {
            property_information.emplace_back("ve_stress_xx_old",1);
            property_information.emplace_back("ve_stress_yy_old",1);

            if (dim == 2)
              {
                property_information.emplace_back("ve_stress_xy_old",1);
              }
            else if (dim == 3)
              {
                property_information.emplace_back("ve_stress_zz_old",1);
                property_information.emplace_back("ve_stress_xy_old",1);
                property_information.emplace_back("ve_stress_xz_old",1);
                property_information.emplace_back("ve_stress_yz_old",1);
              }
          }

        return property_information;
      }



      template <int dim>
      void
      ElasticStress<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Elastic stress");
        {
          prm.declare_entry ("Particle stress value weight", "1.0",
                             Patterns::Double(0.),
                             "The weight given to the value of the stress tensor components "
                             "stored on the particles in the weighted average of those "
                             "values and the values of the compositional fields evaluated on the "
                             "particle location. The average is used in the Material Model inputs "
                             "used to compute the reaction rates and to update the particle property "
                             "with the reaction rates. In some cases, using the field values "
                             "leads to more stable results.");

        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      ElasticStress<dim>::parse_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Elastic stress");
        {
          particle_weight = prm.get_double("Particle stress value weight");
        }
        prm.leave_subsection();
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(ElasticStress,
                                        "elastic stress",
                                        "A plugin in which the particle property tensor is "
                                        "defined as the total elastic stress a particle has "
                                        "accumulated. This plugin modifies the properties "
                                        "with the name ve_stress_*. It first applies the stress "
                                        "change resulting from system evolution during the previous "
                                        "computational timestep, and then the rotation of those "
                                        "stresses into the current timestep. "
                                        "See the viscoelastic or visco_plastic material model "
                                        "documentation for more detailed information.")
    }
  }
}
