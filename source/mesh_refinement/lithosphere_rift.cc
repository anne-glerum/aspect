/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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



#include <aspect/mesh_refinement/lithosphere_rift.h>
#include <aspect/utilities.h>
#include <aspect/initial_composition/lithosphere_rift.h>
#include <aspect/initial_temperature/lithosphere_rift.h>
#include <aspect/geometry_model/box.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <math.h>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    void
    LithosphereRift<dim>::
    initialize ()
    {
    // Check that the required initial composition model is used
    const std::vector<std::string> active_initial_composition_models = this->get_initial_composition_manager().get_active_initial_composition_names();
    AssertThrow(std::find(active_initial_composition_models.begin(),active_initial_composition_models.end(), "lithosphere with rift") != active_initial_composition_models.end(),
                ExcMessage("The lithosphere with rift initial mesh refinement plugin requires the lithosphere with rift initial composition plugin."));

    // Check that the required initial temperature model is used
    const std::vector<std::string> active_initial_temperature_models = this->get_initial_temperature_manager().get_active_initial_temperature_names();
    AssertThrow(std::find(active_initial_temperature_models.begin(),active_initial_temperature_models.end(), "lithosphere with rift") != active_initial_temperature_models.end(),
                ExcMessage("The lithosphere with rift initial mesh refinement plugin requires the lithosphere with rift initial temperature plugin."));
    }


    template <int dim>
    void
    LithosphereRift<dim>::tag_additional_cells () const
    {
      for (typename Triangulation<dim>::active_cell_iterator
           cell = this->get_triangulation().begin_active();
           cell != this->get_triangulation().end(); ++cell)
        {
          if (cell->is_locally_owned())
            {
              bool refine = false;
              bool clear_coarsen = false;

              for ( unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;  ++v)
                {
                  const Point<dim> vertex = cell->vertex(v);

                  const double depth = this->get_geometry_model().depth(vertex);

                  // Determine coordinate system
                  const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;
                  double distance_to_rift_axis = 1e23;
                  Point<2> surface_position;
                  const std::list<std_cxx11::shared_ptr<InitialComposition::Interface<dim> > > initial_composition_objects = this->get_initial_composition_manager().get_active_initial_composition_conditions();
                  for (typename std::list<std_cxx11::shared_ptr<InitialComposition::Interface<dim> > >::const_iterator it = initial_composition_objects.begin(); it != initial_composition_objects.end(); ++it)
                    if( InitialComposition::LithosphereRift<dim> *ic = dynamic_cast<InitialComposition::LithosphereRift<dim> *> ((*it).get()))
                      {
                        surface_position = ic->surface_position(vertex, cartesian_geometry);
                        distance_to_rift_axis = ic->distance_to_rift(surface_position);
                      }

                  if(depth <= reference_crustal_thickness && std::abs(distance_to_rift_axis)<=3.*sigma)
                    {
                      if (cell->level() <= rint(rift_refinement_level))
                        clear_coarsen = true;
                      if (cell->level() <  rint(rift_refinement_level))
                        {
                          refine = true;
                          break;
                        }
                    }
                }

              if (clear_coarsen)
                cell->clear_coarsen_flag ();
              if (refine)
                cell->set_refine_flag ();
            }
        }
    }

    template <int dim>
    void
    LithosphereRift<dim>::
    declare_parameters (ParameterHandler &)
    {

    }

    template <int dim>
    void
    LithosphereRift<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {
        //compute maximum refinement level (initial global + initial adaptive)
        // and set this as minimum for rift area
        rift_refinement_level = prm.get_integer ("Initial global refinement") + prm.get_integer ("Initial adaptive refinement");
      }
      prm.leave_subsection();


      // get sigma and reference crustal thickness
      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("Lithosphere with rift");
        {
          sigma                = prm.get_double ("Standard deviation of Gaussian noise amplitude distribution");
          std::vector<double> thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Layer thicknesses"))),
                                                                                    3,
                                                                                    "Layer thicknesses");
          reference_crustal_thickness = thicknesses[0]+thicknesses[1]+thicknesses[2];
        }
        prm.leave_subsection();
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
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(LithosphereRift,
                                              "lithosphere with rift",
                                              "A mesh refinement criterion that ensures a "
                                              "minimum refinement level described by an "
                                              "explicit formula with the depth or position "
                                              "as argument. Which coordinate representation "
                                              "is used is determined by an input parameter. "
                                              "Whatever the coordinate system chosen, the "
                                              "function you provide in the input file will "
                                              "by default depend on variables `x', `y' and "
                                              "`z' (if in 3d). However, the meaning of these "
                                              "symbols depends on the coordinate system. In "
                                              "the Cartesian coordinate system, they simply "
                                              "refer to their natural meaning. If you have "
                                              "selected `depth' for the coordinate system, "
                                              "then `x' refers to the depth variable and `y' "
                                              "and `z' will simply always be zero. If you "
                                              "have selected a spherical coordinate system, "
                                              "then `x' will refer to the radial distance of "
                                              "the point to the origin, `y' to the azimuth "
                                              "angle and `z' to the polar angle measured "
                                              "positive from the north pole. Note that the "
                                              "order of spherical coordinates is r,phi,theta "
                                              "and not r,theta,phi, since this allows for "
                                              "dimension independent expressions. "
                                              "Each coordinate system also includes a final `t' "
                                              "variable which represents the model time, evaluated "
                                              "in years if the 'Use years in output instead of seconds' "
                                              "parameter is set, otherwise evaluated in seconds. "
                                              "After evaluating the function, its values are "
                                              "rounded to the nearest integer."
                                              "\n\n"
                                              "The format of these "
                                              "functions follows the syntax understood by the "
                                              "muparser library, see Section~\\ref{sec:muparser-format}.")
  }
}
