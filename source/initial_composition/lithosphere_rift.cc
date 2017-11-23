/*
  Copyright (C) 2017 by the authors of the ASPECT code.

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


#include <aspect/initial_composition/lithosphere_rift.h>
#include <aspect/initial_temperature/lithosphere_rift.h>
#include <aspect/geometry_model/box.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    void
    LithosphereRift<dim>::
    initialize ()
    {
      // Check that the required initial composition model is used
      const std::vector<std::string> active_initial_temperature_models = this->get_initial_temperature_manager().get_active_initial_temperature_names();
      AssertThrow(find(active_initial_temperature_models.begin(),active_initial_temperature_models.end(), "lithosphere with rift") != active_initial_temperature_models.end(),
                  ExcMessage("The lithosphere with rift initial composition plugin requires the lithosphere with rift initial temperature plugin."));
    }

    template <int dim>
    double
    LithosphereRift<dim>::
    initial_composition (const Point<dim> &position,
                         const unsigned int compositional_index) const
    {
      // Retrieve the indices of the fields that represent the lithospheric layers.
      // We assume a 3-layer system with an upper crust, lower crust and lithospheric mantle
      const unsigned int id_upper = this->introspection().compositional_index_for_name("upper");
      const unsigned int id_lower = this->introspection().compositional_index_for_name("lower");
      const unsigned int id_mantle_L = this->introspection().compositional_index_for_name("mantle_L");

      // Determine coordinate system
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;

      const Point<2> surface_point = surface_position(position, cartesian_geometry);

      // Get the distance to the line segments along a path parallel to the surface
      const double distance_to_rift_axis = distance_to_rift(surface_point);

      // Get the signed distance to a polygon of different lithospheric thicknesses
      const double distance_to_L_polygon = distance_to_polygon(surface_point);

      // Compute the local thickness of the upper crust, lower crust and mantle part of the lithosphere
      // (in this exact order) based on the distance from the rift axis.
      const double local_upper_crust_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon/sigma))*polygon_thicknesses[0]+(0.5-0.5*std::tanh(distance_to_L_polygon/sigma))*thicknesses[0])*
                                                          (1.0 - A[0] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))));
      const double local_lower_crust_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon/sigma))*polygon_thicknesses[1]+(0.5-0.5*std::tanh(distance_to_L_polygon/sigma))*thicknesses[1])*
                                                          (1.0 - A[1] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))));
      const double local_mantle_lithosphere_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon/sigma))*polygon_thicknesses[2]+(0.5-0.5*std::tanh(distance_to_L_polygon/sigma))*thicknesses[2])*
                                                                 (1.0 - A[2] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma,2)))));

      // Compute depth
      const double depth = this->get_geometry_model().depth(position);

      // Check which layer we're in and return value.
      if (depth <= local_upper_crust_thickness && compositional_index == id_upper)
        return 1.;
      else if (depth > local_upper_crust_thickness && depth <= local_upper_crust_thickness+local_lower_crust_thickness && compositional_index == id_lower)
        return 1.;
      else if (depth > local_upper_crust_thickness+local_lower_crust_thickness && depth <= local_upper_crust_thickness+local_lower_crust_thickness+local_mantle_lithosphere_thickness && compositional_index == id_mantle_L)
        return 1.;
      else
        return 0.;
    }

    template <int dim>
    double
    LithosphereRift<dim>::
    distance_to_rift (const Point<2> &surface_position) const
    {
      // Initiate distance with large value
      double distance_to_rift_axis = 1e23;
      double temp_distance = 0;

      // Loop over all line segments
      for (unsigned int i_segments = 0; i_segments < point_list.size(); ++i_segments)
        {
          temp_distance = (dim == 2) ? std::abs(surface_position[0]-point_list[i_segments][0][0]) : std::abs(Utilities::distance_to_line<dim>(point_list[i_segments], surface_position));

          // Get the minimum distance
          distance_to_rift_axis = std::min(distance_to_rift_axis, temp_distance);
        }

      return distance_to_rift_axis;
    }

    template <int dim>
    Point<2>
    LithosphereRift<dim>::
    surface_position (const Point<dim> &position,
                      const bool cartesian_geometry) const
    {
      // When in 2d, the second coordinate is zero
      Point<2> surface_point;
      if (cartesian_geometry)
        {
          for(unsigned int d=0; d<dim-1; ++d)
            surface_point[d]=position[d];
        }
      // chunk (spherical) geometries
      else
        {
          // spherical coordinates in radius [m], lon [rad], colat [rad] format
          const std_cxx11::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
          // return lon [degrees], lat [degrees]
          for (unsigned int d=0; d<dim-1; ++d)
            surface_point[d] = spherical_point[d+1]*180./numbers::PI;
          if (dim == 3)
            surface_point[dim-2] = 90. - surface_point[dim-2];
        }

      return surface_point;
    }

    template <int dim>
    double
    LithosphereRift<dim>::
    distance_to_polygon (const Point<2> &surface_position) const
    {
      return  polygon_point_list.size()>0 ? Utilities::signed_distance_to_polygon<dim>(polygon_point_list, surface_position) : -1e23;
    }

    template <int dim>
    void
    LithosphereRift<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("Lithosphere with rift");
        {
          prm.declare_entry ("Standard deviation of Gaussian rift geometry", "20000",
                             Patterns::Double (0),
                             "The standard deviation of the Gaussian distribution of the amplitude of the strain noise. "
                             "Note that this parameter is taken to be the same for all rift segments. "
                             "Units: $m$ or degrees.");
          prm.declare_entry ("Amplitude of Gaussian rift geometry", "0.2",
                             Patterns::List(Patterns::Double (-1,1)),
                             "The amplitude of the Gaussian distribution of the amplitude of the strain noise. "
                             "Note that this parameter is taken to be the same for all rift segments, but can "
                             "vary per lithosphere layer. "
                             "Units: none.");
          prm.declare_entry ("Layer thicknesses", "30000.",
                             Patterns::List(Patterns::Double(0)),
                             "List of thicknesses for the bottom of the lithospheric layers,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value.  Units: $m$");
          prm.declare_entry ("Rift axis line segments",
                             "",
                             Patterns::Anything(),
                             "Set the line segments that represent the rift axis. Each segment is made up of "
                             "two points that represent horizontal coordinates (x,y) or (lon,lat). "
                             "The exact format for the point list describing the segments is "
                             "\"x1,y1>x2,y2;x2,y2>x3,y3;x4,y4>x5,y5\". Note that the segments can be connected "
                             "or isolated. The units of the coordinates are "
                             "dependent on the geometry model. In the box model they are in meters, in the "
                             "chunks they are in degrees.");
          prm.declare_entry ("Lithospheric polygon",
                             "",
                             Patterns::Anything(),
                             "Set the polygon that represents an area of different lithospheric thickness. "
                             "The polygon is a list of "
                             "points that represent horizontal coordinates (x,y) or (lon,lat). "
                             "The exact format for the point list describing the polygon is "
                             "\"x1,y1;x2,y2;x3,y3;x4,y4;x5,y5\". Note that the polygon is assumed to be closed. "
                             "The units of the coordinates are "
                             "dependent on the geometry model. In the box model they are in meters, in the "
                             "chunks they are in degrees.");
          prm.declare_entry ("Lithospheric polygon layer thicknesses", "30000.",
                             Patterns::List(Patterns::Double(0)),
                             "List of thicknesses for the bottom of the lithospheric layers,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value.  Units: $m$");

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    LithosphereRift<dim>::parse_parameters (ParameterHandler &prm)
    {

      // Check that the required compositional fields exist.
      AssertThrow(this->introspection().compositional_name_exists("upper"),ExcMessage("We need a compositional field called 'upper' representing the upper crust."));
      AssertThrow(this->introspection().compositional_name_exists("lower"),ExcMessage("We need a compositional field called 'lower' representing the lower crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));

      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("Lithosphere with rift");
        {
          sigma                = prm.get_double ("Standard deviation of Gaussian rift geometry");
          A                    = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Amplitude of Gaussian rift geometry"))),
                                                                         3,
                                                                         "Amplitude of Gaussian rift geometry");
          thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Layer thicknesses"))),
                                                              3,
                                                              "Layer thicknesses");
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
                  //if (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) == NULL)
                  //   point_list[i_segment][i_points] = (Point<2>(temp_point[0]/180.*numbers::PI, 0.5*numbers::PI-temp_point[1]/180.*numbers::PI));
                  //else
                     point_list[i_segment][i_points] = (Point<2>(temp_point[0], temp_point[1]));
                }
            }
          // Read in the string of polygon points
          const std::string temp_all_points = prm.get("Lithospheric polygon");
          // Split the string into point strings
          const std::vector<std::string> temp_points = Utilities::split_string_list(temp_all_points,';');
          const unsigned int n_temp_points = temp_points.size();
          polygon_point_list.resize(n_temp_points);
          // Loop over the points of the polygon. Each point should consist of 2 values (lon and lat coordinate).
          if(dim == 3 && n_temp_points != 0)
            {
              AssertThrow(n_temp_points>=3, ExcMessage ("The number of polygon points should be equal or larger than 3."));
              for (unsigned int i_points = 0; i_points < n_temp_points; i_points++)
                {
                  const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_points[i_points],','));
                  Assert(temp_point.size() == dim-1,ExcMessage ("The given coordinates of point '" + temp_points[i_points] + "' are not correct. "
                                                                "It should only contain 2 parts: "
                                                                "the longitude and latitude, separated by a ','."));

                  // Add the point to the list of points for this segment
                  //if (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) == NULL)
                  //   polygon_point_list[i_points] = (Point<2>(temp_point[0]/180.*numbers::PI, 0.5*numbers::PI-temp_point[1]/180.*numbers::PI));
                  //else
                     polygon_point_list[i_points] = (Point<2>(temp_point[0], temp_point[1]));
                }
            }

          if(n_temp_points == 0)
            polygon_thicknesses = thicknesses;
          else
            polygon_thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Lithospheric polygon layer thicknesses"))),
                                                                          3,
                                                                          "Lithospheric polygon layer thicknesses");


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
  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(LithosphereRift,
                                              "lithosphere with rift",
                                              "A class that implements initial conditions for the porosity field "
                                              "by computing the equilibrium melt fraction for the given initial "
                                              "condition and reference pressure profile. Note that this plugin only "
                                              "works if there is a compositional field called `porosity', and the "
                                              "used material model implements the 'MeltFractionModel' interface. "
                                              "For all compositional fields except porosity this plugin returns 0.0, "
                                              "and they are therefore not changed as long as the default `add' "
                                              "operator is selected for this plugin.")
  }
}
