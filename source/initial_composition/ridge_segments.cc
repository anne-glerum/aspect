/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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


#include <aspect/initial_composition/ridge_segments.h>
#include <aspect/postprocess/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/material_model/visco_plastic.h>

namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    RidgeSegments<dim>::RidgeSegments ()
    {}

    template <int dim>
    void
    RidgeSegments<dim>::
    initialize ()
    {
      // Check that the required initial temperature model is used
      const std::vector<std::string> active_initial_temperature_models = this->get_initial_temperature_manager().get_active_initial_temperature_names();
      AssertThrow(find(active_initial_temperature_models.begin(),active_initial_temperature_models.end(), "ridge segments") != active_initial_temperature_models.end(),
                  ExcMessage("The ridge segments initial composition plugin requires the lithosphere with ridge initial temperature plugin."));

      // Check that the required material model ("visco plastic") is used
      AssertThrow((dynamic_cast<MaterialModel::ViscoPlastic<dim> *> (const_cast<MaterialModel::Interface<dim> *>(&this->get_material_model()))) != 0,
                  ExcMessage("The lithosphere with rift initial temperature plugin requires the viscoplastic material model plugin."));

      if (this->convert_output_to_years())
        spreading_velocity /= year_in_seconds;
    }

    template <int dim>
    double
    RidgeSegments<dim>::
    initial_composition (const Point<dim> &position, const unsigned int n_comp) const
    {
      // Determine coordinate system
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;

      // Get the surface position and the distance to the ridge
      Point<2> surface_position = surface_position(position, cartesian_geometry);
      const double distance_to_ridge = distance_to_ridge(surface_position, cartesian_geometry);

      // The depth with respect to the initial model surface
      const double depth = this->get_geometry_model().depth(position);

      // Determine plate age based on distance from domain boundary
      const double plate_age = 0.5 * spreading_velocity * distance_to_ridge;

      // The plate thickness
      const double plate_thickness = 2.32*std::sqrt(thermal_diffusivity*plate_age));

      // The crust is of uniform thickness, but take the temperature discontinuity
      // at the mid oceanic ridge into account.
      if (depth <= std::min(crustal_thickness, plate_thickness) && n_comp == id_crust)
        return 1.;
        else if (depth <= plate_thickness && n_comp == id_mantle_L)
        return 1.;
        else
          return 0.;
        }

  template <int dim>
  double
  RidgeSegments<dim>::
  distance_to_ridge (const Point<2> &surface_position,
                     const bool cartesian_geometry) const
    {
      // Initiate distance with large value
      double distance_to_rift_axis = 1e23;
      double temp_distance = 0;

      // Loop over all line segments
      // TODO: fix stupid dim of surface_position
      for (unsigned int i_segments = 0; i_segments < point_list.size(); ++i_segments)
        {
          if (cartesian_geometry)
            {
              if (dim == 2)
                temp_distance = std::abs(surface_position[0]-point_list[i_segments][0][0]);
              else
                temp_distance = std::abs(Utilities::distance_to_line<dim>(point_list[i_segments], surface_position));
            }
          // chunk (spherical) geometries
          else
            temp_distance = (dim == 2) ? std::abs(surface_position[0]-point_list[i_segments][0][0]) : Utilities::distance_to_line<dim>(point_list[i_segments], surface_position);

          // Get the minimum distance
          distance_to_rift_axis = std::min(distance_to_rift_axis, temp_distance);
        }

      return distance_to_rift_axis;
    }

    template <int dim>
    Point<2>
    RidgeSegments<dim>::
    surface_position (const Point<dim> &position,
                      const bool cartesian_geometry) const
    {
      // When in 2d, the second coordinate is zero
      Point<2> surface_point;
      if (cartesian_geometry)
        {
          for (unsigned int d=0; d<dim-1; ++d)
            surface_point[d]=position[d];
        }
      // chunk (spherical) geometries
      else
        {
          // spherical coordinates in radius [m], lon [rad], colat [rad] format
          const std_cxx11::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
          for (unsigned int d=0; d<dim-1; ++d)
            surface_point[d] = spherical_point[d+1];
        }

      return surface_point;
    }

    template <int dim>
    void
    RidgeSegments<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Plate cooling");
        {
          prm.declare_entry ("Crustal thickness", "6e3",
                             Patterns::Double (0),
                             "The uniform thickness of the oceanic crust. "
                             "Unit: m.");
          prm.declare_entry ("Ridge line segments",
                             "",
                             Patterns::Anything(),
                             "Set the line segments that represent the rift axis. Each segment is made up of "
                             "two points that represent horizontal coordinates (x,y) or (lon,lat). "
                             "The exact format for the point list describing the segments is "
                             "\"x1,y1>x2,y2;x2,y2>x3,y3;x4,y4>x5,y5\". Note that the segments can be connected "
                             "or isolated. The units of the coordinates are "
                             "dependent on the geometry model. In the box model they are in meters, in the "
                             "chunks they are in degrees.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    RidgeSegments<dim>::parse_parameters (ParameterHandler &prm)
    {
      // we need to get at the number of compositional fields here to
      // initialize the function parser. unfortunately, we can't get it
      // via SimulatorAccess from the simulator itself because at the
      // current point the SimulatorAccess hasn't been initialized
      // yet. so get it from the parameter file directly.
      prm.enter_subsection ("Compositional fields");
      const unsigned int n_compositional_fields = prm.get_integer ("Number of fields");
      prm.leave_subsection ();

      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("Plate cooling");
        {
          spreading_velocity = prm.get_double ("Spreading velocity");
          max_plate_thickness = prm.get_double ("Maximum oceanic plate thickness");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Plate cooling");
        {
          crustal_thickness = prm.get_double ("Crustal thickness");
          // Read in the string of segments
          const std::string temp_all_segments = prm.get("Rift axis line segments");
          // Split the string into segment strings
          const std::vector<std::string> temp_segments = Utilities::split_string_list(temp_all_segments,';');
          const unsigned int n_temp_segments = temp_segments.size();
          point_list.resize(n_temp_segments);
          // Loop over the segments to extract the points
          for (unsigned int i_segment = 0; i_segment < n_temp_segments; i_segment++)
            {
              // In 3d a line segment consists of 2 points,
              // in 2d only 1 (ridge axis orthogonal two x and y)
              point_list[i_segment].resize(dim-1);


              const std::vector<std::string> temp_segment = Utilities::split_string_list(temp_segments[i_segment],'>');

              if (dim == 3)
                {
                  Assert(temp_segment.size() == 2,ExcMessage ("The given coordinate '" + temp_segment[i_segment] + "' is not correct. "
                                                              "It should only contain 2 parts: "
                                                              "the two points of the segment, separated by a '>'."));
                }
              else
                {
                  Assert(temp_segment.size() == 1,ExcMessage ("The given coordinate '" + temp_segment[i_segment] + "' is not correct. "
                                                              "In 2d it should only contain only 1 part: "));
                }

              // Loop over the dim-1 points of each segment (i.e. in 2d only 1 point is required for a 'segment')
              for (unsigned int i_points = 0; i_points < dim-1; i_points++)
                {
                  const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_segment[i_points],','));
                  Assert(temp_point.size() == 2,ExcMessage ("The given coordinates of segment '" + temp_segment[i_points] + "' are not correct. "
                                                            "It should only contain 2 parts: "
                                                            "the two coordinates of the segment end point, separated by a ','."));

                  // Add the point to the list of points for this segment
                  if (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) == NULL)
                    point_list[i_segment][i_points] = (Point<2>(temp_point[0]/180.*numbers::PI, 0.5*numbers::PI-temp_point[1]/180.*numbers::PI));
                  else
                    point_list[i_segment][i_points] = (Point<2>(temp_point[0], temp_point[1]));
                }
            }
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      AssertThrow(this->introspection().compositional_name_exists("crust"),
                  ExcMessage("We need a compositional field called 'crust' representing the oceanic crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),
                  ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));
      id_mantle_L = this->introspection().compositional_index_for_name("mantle_L");
      id_crust = this->introspection().compositional_index_for_name("crust");

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Visco Plastic");
        {
          const std::vector<double> temp_thermal_diffusivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal diffusivities"))),
                                                                 n_fields+1,
                                                                 "Thermal diffusivities");

          // Assume the mantle lithosphere diffusivity is representative for whole plate
          thermal_diffusivity = temp_thermal_diffusivities[id_mantle_L+1];
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
  namespace CompositionalInitialConditions
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(RidgeSegments,
                                              "ridge segments",
                                              "The compositions are based on the cooling half space model "
                                              "and the distance to the ridge segments. ")
  }
}
