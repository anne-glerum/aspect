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

#include <aspect/particle/property/test_particle_update.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
//      template <int dim>
//      TestParticleUpdate<dim>::TestParticleUpdate ()
//        :
//      {}



      template <int dim>
      void
      TestParticleUpdate<dim>::initialize ()
      {
        this->get_signals().post_restore_particles.connect(
          [&](typename Particle::World<dim> &particle_world)
        {
          this->update_particles(particle_world);
        }
        );
      }



      template <int dim>
      void
      TestParticleUpdate<dim>::update_particles(typename Particle::World<dim> &particle_world) const
      {
        // Determine the data position of the 'test' property
        //const Manager<dim> *particle_property_manager = &particle_world.get_property_manager();
        //const unsigned int data_position = particle_property_manager.get_property_component_by_name("test");
        const unsigned int data_position = particle_world.get_property_manager().get_data_info().get_position_by_field_name("test");

        // Get handler
        Particle::ParticleHandler<dim> &particle_handler = particle_world.get_particle_handler();

        // Loop over all cells and update the particles cell-wise
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              typename ParticleHandler<dim>::particle_iterator_range
              particles_in_cell = particle_handler.particles_in_cell(cell);

              // Only update particles, if there are any in this cell
              if (particles_in_cell.begin() != particles_in_cell.end())
                {
                  for (auto particle = particles_in_cell.begin(); particle!=particles_in_cell.end(); ++particle)
                    {
                      // Update the property test with 0.5
                      particle->get_properties()[data_position] += 0.5;
                    }
                }
            }
      }



      template <int dim>
      void
      TestParticleUpdate<dim>::initialize_one_particle_property(const Point<dim> &,
                                                                std::vector<double> &data) const
      {
        // Start with a value of 10
        data.push_back(10);
      }



      template <int dim>
      void
      TestParticleUpdate<dim>::update_particle_property(const unsigned int ,
                                                        const Vector<double> &,
                                                        const std::vector<Tensor<1,dim>> &,
                                                        typename ParticleHandler<dim>::particle_iterator &) const
      {
      }



      template <int dim>
      UpdateTimeFlags
      TestParticleUpdate<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      UpdateFlags
      TestParticleUpdate<dim>::get_needed_update_flags () const
      {
        return update_values;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      TestParticleUpdate<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;

        property_information.emplace_back("test",1);

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
      ASPECT_REGISTER_PARTICLE_PROPERTY(TestParticleUpdate,
                                        "test particle update",
                                        "A plugin in which the particle property 'test' is "
                                        "only updated each timestep by a constant value of 0.5. "
                                        "This update occurs through the signal 'post_restore_particles' "
                                        "that is triggered after particles are restored at the beginning "
                                        "of each nonlinear iteration in an iterative Advection solver scheme.")

    }
  }
}
