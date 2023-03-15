/*
  Copyright (C) 2015 - 2022 by the authors of the ASPECT code.

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
        property_position(0)
      {}



      template <int dim>
      void
      ElasticStress<dim>::initialize ()
      {
        material_inputs = MaterialModel::MaterialModelInputs<dim>(1, this->n_compositional_fields());

        material_outputs = MaterialModel::MaterialModelOutputs<dim>(1, this->n_compositional_fields());

        AssertThrow((Plugins::plugin_type_matches<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model())
                     ||
                     Plugins::plugin_type_matches<const MaterialModel::Viscoelastic<dim>>(this->get_material_model())),
                    ExcMessage("This particle property only makes sense in combination with the viscoelastic or visco_plastic material model."));

        AssertThrow(this->get_parameters().enable_elasticity == true,
                    ExcMessage ("This particle property should only be used if 'Enable elasticity' is set to true"));

        // TODO more elegant fix
        const auto &property_information = this->get_particle_world().get_property_manager().get_data_info();
        property_position = property_information.get_position_by_field_name("ve_stress_xx");
      }



      template <int dim>
      void
      ElasticStress<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                           std::vector<double> &data) const
      {
        // // Give each elastic stress field its initial composition if one is prescribed.
        // data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xx")));

        // data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_yy")));

        // if (dim == 2)
        //   {
        //     data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xy")));
        //   }
        // else if (dim == 3)
        //   {
        //     data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_zz")));

        //     data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xy")));

        //     data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_xz")));

        //     data.push_back(this->get_initial_composition_manager().initial_composition(position,this->introspection().compositional_index_for_name("ve_stress_yz")));
        //   }

      }



      template <int dim>
      void
      ElasticStress<dim>::update_particle_property(const unsigned int data_position,
                                                   const Vector<double> &solution,
                                                   const std::vector<Tensor<1,dim>> &gradients,
                                                   typename ParticleHandler<dim>::particle_iterator &particle) const
      {
        material_inputs.position[0] = particle->get_location();

#if DEAL_II_VERSION_GTE(9,4,0)
        material_inputs.current_cell = typename DoFHandler<dim>::active_cell_iterator(*particle->get_surrounding_cell(),
                                                                                      &(this->get_dof_handler()));
#else
        material_inputs.current_cell = typename DoFHandler<dim>::active_cell_iterator(*particle->get_surrounding_cell(this->get_triangulation()),
                                                                                      &(this->get_dof_handler()));
#endif
        material_inputs.temperature[0] = solution[this->introspection().component_indices.temperature];

        material_inputs.pressure[0] = solution[this->introspection().component_indices.pressure];

        for (unsigned int d = 0; d < dim; ++d)
          material_inputs.velocity[0][d] = solution[this->introspection().component_indices.velocities[d]];

        for (unsigned int n = 0; n < this->n_compositional_fields(); ++n)
          material_inputs.composition[0][n] = solution[this->introspection().component_indices.compositional_fields[n]];

        Tensor<2,dim> grad_u;
        for (unsigned int d=0; d<dim; ++d)
          grad_u[d] = gradients[d];
        material_inputs.strain_rate[0] = symmetrize (grad_u);

        this->get_material_model().evaluate (material_inputs,material_outputs);


        for (unsigned int i = 0; i < SymmetricTensor<2,dim>::n_independent_components ; ++i)
          // TODO provide more elegant fix
          // Instead of using the data_position given as input to this function,
          // we use the index of the compositional field representing the first stress tensor component.
          // Each particle property plugin gets its own data position, and when iterating through the
          // plugins, the data position is moved forward each time. However, here we want to apply
          // an update to properties that have already been updated by another plugin. I.e., we
          // want to update the 'composition' properties, which carry the stress tensor components.
          // In the current plugin however, new 've_stress_*' properties are created and updated. 
          // The creation of new properties can be removed when we assert that the 'composition' property
          // plugin is used. This plugin is necessary because it updates the composition properties with the 
          // newest solution, i.e. the one including the operator splitting update.
          // The current plugin then applies the reaction term (the rotation). We could also include the
          // solution update in this plugin and assert that the 'composition' plugin is not active.
          // For now, as the least invasive hacky fix, I do not create new ve_stress_* properties and 
          // ask for the data position of the particle property 've_stress_xx', which is created by the
          // 'composition' property plugin.
          particle->get_properties()[property_position + i] += material_outputs.reaction_terms[0][i];
      }



      template <int dim>
      UpdateTimeFlags
      ElasticStress<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      UpdateFlags
      ElasticStress<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      ElasticStress<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;

        // //Check which fields are used in model and make an output for each.
        // if (this->introspection().compositional_name_exists("ve_stress_xx"))
        //   property_information.emplace_back("ve_stress_xx",1);

        // if (this->introspection().compositional_name_exists("ve_stress_yy"))
        //   property_information.emplace_back("ve_stress_yy",1);

        // if (dim == 2)
        //   {
        //     if (this->introspection().compositional_name_exists("ve_stress_xy"))
        //       property_information.emplace_back("ve_stress_xy",1);
        //   }
        // else if (dim == 3)
        //   {
        //     if (this->introspection().compositional_name_exists("ve_stress_zz"))
        //       property_information.emplace_back("ve_stress_zz",1);

        //     if (this->introspection().compositional_name_exists("ve_stress_xy"))
        //       property_information.emplace_back("ve_stress_xy",1);

        //     if (this->introspection().compositional_name_exists("ve_stress_xz"))
        //       property_information.emplace_back("ve_stress_xz",1);

        //     if (this->introspection().compositional_name_exists("ve_stress_yz"))
        //       property_information.emplace_back("ve_stress_yz",1);
        //   }

        return property_information;
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
                                        "accumulated. See the viscoelastic material model "
                                        "documentation for more detailed information.")

    }
  }
}
