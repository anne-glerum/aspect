/*
  Copyright (C) 2014 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/



#include <aspect/mesh_refinement/plume.h>
#include <aspect/boundary_temperature/plume_only.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/ellipsoidal_chunk.h>


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <math.h>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    void
    Plume<dim>::tag_additional_cells () const
    {
      // verify that the we have a plume boundary temperature model
      // which will give us the plume position
      Assert (this->get_boundary_temperature_manager().template has_matching_boundary_temperature_model<BoundaryTemperature::PlumeOnly<dim> >(),
              ExcMessage ("This refinement parameter is only implemented if the boundary "
                          "temperature plugin is the 'plume' model."));

      const BoundaryTemperature::PlumeOnly<dim> &boundary_temperature =
        this->get_boundary_temperature_manager().template get_matching_boundary_temperature_model<BoundaryTemperature::PlumeOnly<dim> >();

      // TODO once the plume head is inside the domain,
      // there is no knowing how fast it travels. It could well
      // be slower than the prescribed inflow velocity.
      // In this case, the mesh refinement criteria specified
      // here do not follow the plume head correctly.
      // Perhaps after the plume head has passed, assume the tail velocity?
      // TODO also, the plume head does not necessarily move laterally as the
      // plume tail does at the boundary, so I'm not sure how to use this plugin
      // Perhaps only for the bottom boundary?
      const Point<dim> plume_position = boundary_temperature.get_plume_position();
      const double distance_head_to_boundary = head_velocity * (this->get_time() - model_time_to_start_plume_tail);
      Point<dim> top_tail_cylinder_axis = plume_position;
      if (dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model()) != 0 )
        {
          top_tail_cylinder_axis -= plume_position;
          top_tail_cylinder_axis[dim-1] += distance_head_to_boundary;
        }
      else if (GeometryModel::Chunk<dim> *gm = dynamic_cast<GeometryModel::Chunk<dim> *>
                                               (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        top_tail_cylinder_axis *= (gm->inner_radius()+distance_head_to_boundary) / plume_position.norm();
      else if (GeometryModel::EllipsoidalChunk<dim> *gm = dynamic_cast<GeometryModel::EllipsoidalChunk<dim> *>
                                                          (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          AssertThrow(gm->get_eccentricity()==0, ExcMessage("This plume boundary velocity plugin does not work for an ellipsoidal domain."));
          const double outer_radius = gm->get_semi_major_axis_a();
          top_tail_cylinder_axis *= (outer_radius - gm->maximal_depth() + distance_head_to_boundary) / plume_position.norm();
        }
      else
        AssertThrow(false, ExcNotImplemented());

      const double c2 = top_tail_cylinder_axis * top_tail_cylinder_axis;


      for (typename Triangulation<dim>::active_cell_iterator
           cell = this->get_triangulation().begin_active();
           cell != this->get_triangulation().end(); ++cell)
        {
          if (cell->is_locally_owned())
            {
              bool refine = false;
              bool clear_coarsen = false;
              bool coarsen = false;
              bool clear_refine = false;

              for ( unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;  ++v)
                {
                  Point<dim> vertex = cell->vertex(v);

                  // Find the head
                  unsigned int minimum_refinement_level = std::max(overall_minimum_refinement_level,
                                                                   ((vertex - plume_position).norm() < plume_refinement_radius)
                                                                   ?
                                                                   plume_refinement_level
                                                                   :
                                                                   0);
                  // Find the plume tail
                  const double c1 = top_tail_cylinder_axis * vertex;
                  double distance_to_tail_axis = std::numeric_limits<double>::max();
                  if (c1>0 && c2>c1)
                    distance_to_tail_axis = Tensor<1,dim> (vertex -  c1/c2 * top_tail_cylinder_axis).norm();
                  minimum_refinement_level = std::max(minimum_refinement_level,
                                                      (distance_to_tail_axis <= tail_radius + (plume_refinement_radius-head_radius) ?
                                                       plume_refinement_level
                                                       :
                                                       0));


                  if (cell->level() == rint(minimum_refinement_level))
                    {
                      clear_coarsen = true;
                      clear_refine = true;
                      refine = false;
                      coarsen = false;
                    }
                  if (cell->level() <  rint(minimum_refinement_level))
                    {
                      clear_coarsen = true;
                      refine = true;
                      clear_refine = false;
                      coarsen = false;
                      break;
                    }
                  if (cell->level() > rint(minimum_refinement_level))
                    {
                      clear_refine = true;
                      coarsen = true;
                      clear_coarsen = false;
                      refine = false;
                    }
                }

              if (clear_coarsen)
                cell->clear_coarsen_flag ();
              if (refine)
                cell->set_refine_flag ();
              if (clear_refine)
                cell->clear_refine_flag ();
              if (coarsen)
                cell->set_coarsen_flag ();
            }
        }
    }

    template <int dim>
    void
    Plume<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Plume");
      {
        prm.enter_subsection("Plume refinement");
        {
          prm.declare_entry ("Plume refinement level", "4",
                             Patterns::Integer (0),
                             "Minimum refinement level within the given "
                             "distance from the plume center.");
          prm.declare_entry ("Plume refinement radius", "0",
                             Patterns::Double (0),
                             "Lateral distance from the plume in which the "
                             "plume refinement level is enforced.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    Plume<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Plume");
      {
        tail_radius                    = prm.get_double ("Radius");
        head_radius                    = prm.get_double("Head radius");
        head_velocity                  = prm.get_double("Head velocity");
        model_time_to_start_plume_tail = prm.get_double ("Model time to start plume tail");

        prm.enter_subsection("Plume refinement");
        {
          plume_refinement_level = prm.get_integer ("Plume refinement level");
          plume_refinement_radius = prm.get_double ("Plume refinement radius");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      prm.enter_subsection("Mesh refinement");
      {
        overall_minimum_refinement_level = prm.get_integer ("Minimum refinement level");
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(Plume,
                                              "plume",
                                              "A mesh refinement criterion that ensures a "
                                              "minimum refinement level dependent on the "
                                              "distance to a plume center that is provided "
                                              "time dependent in a file.")
  }
}
