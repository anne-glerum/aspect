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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
 */


#include <aspect/global.h>
#include <aspect/initial_composition/litho1.h>
#include <aspect/geometry_model/box.h>


namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    Litho1<dim>::Litho1 ()
    {}


    template <int dim>
    void
    Litho1<dim>::initialize ()
    {
      // Find the boundary indicators that represents the surface and the bottom of the domain
      surface_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");
      left_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("west");

      // Abuse the top and bottom boundary id to create to tables,
      // one for the crustal thickness, and one for the mantle thickness
      // surface_boundary_id indicates the crust table and 
      // bottom_boundary_id the LAB table
      std::set<types::boundary_id> boundary_set;
      boundary_set.insert(surface_boundary_id);
      boundary_set.insert(bottom_boundary_id);
      boundary_set.insert(left_boundary_id);

      // The input ascii table contains one component, either the crust depth or the LAB depth
      Utilities::AsciiDataBoundary<dim>::initialize(boundary_set,
                                                    1);
    }


    template <int dim>
    double
    Litho1<dim>::
    initial_composition (const Point<dim> &position,
                         const unsigned int n_comp) const
    {
      // The Moho depth is the first component
      const double Moho_depth = Utilities::AsciiDataBoundary<dim>::get_data_component(surface_boundary_id,
                                                                                      position,
                                                                                      0);
      // The LAB depth is the second and third component
      const double LAB_depth_1 = std::max(min_LAB_thickness,
                                        std::max(Moho_depth,
                                                 Utilities::AsciiDataBoundary<dim>::get_data_component(bottom_boundary_id,
                                                                                                       position,
                                                                                                       0)));

      const double LAB_depth_2 = std::max(min_LAB_thickness,
                                        std::max(Moho_depth,
                                                 Utilities::AsciiDataBoundary<dim>::get_data_component(left_boundary_id,
                                                                                                       position,
                                                                                                       0)));

      Point<2> surface_position;
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL;
      if (cartesian_geometry)
        {
          for (unsigned int d=0; d<dim-1; ++d)
            surface_position[d]=position[d];
        }
      // chunk (spherical) geometries
      else
        {
          // spherical coordinates in radius [m], lon [rad], colat [rad] format
          const std_cxx11::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
          // return lon [degrees], lat [degrees]
          for (unsigned int d=0; d<dim-1; ++d)
            surface_position[d] = spherical_point[d+1]*180./numbers::PI;
          if (dim == 3)
            surface_position[dim-2] = 90. - surface_position[dim-2];
        }

      // Inside is positive, outside negative.
      double distance_to_polygon = 0;
      if (dim == 2)
        {
          double sign = -1.;
          if (surface_position[0]>polygon_point_list[0][0] && surface_position[0]<polygon_point_list[1][0])
            sign = 1.;
          distance_to_polygon = sign * std::min(std::abs(polygon_point_list[1][0] - surface_position[0]), std::abs(surface_position[0] - polygon_point_list[0][0]));
        }
      else
        {
          distance_to_polygon = Utilities::signed_distance_to_polygon<dim>(polygon_point_list, Point<2>(surface_position[0],surface_position[dim-2]));
        }

      const double LAB_depth =  merge_LAB_grids ? (0.5+0.5*std::tanh(distance_to_polygon/merge_LAB_grids_halfwidth))*LAB_depth_2
                                +(0.5-0.5*std::tanh(distance_to_polygon/merge_LAB_grids_halfwidth))*LAB_depth_1
                                : LAB_depth_1;

      AssertThrow(Moho_depth <= LAB_depth, ExcMessage("The crust is thicker than the LAB."));

      // The depth of the point under investigation
      const double depth = this->get_geometry_model().depth(position);

      // The upper crustal field (with field id 0)
      if (depth < Moho_depth*upper_crust_fraction && n_comp == upper_crust_id)
        return 1.;
      // The lower crustal field (with field id 1 (or 0))
      if (depth >= Moho_depth*upper_crust_fraction && depth < Moho_depth && n_comp == lower_crust_id)
        return 1.;
      // The lithospheric mantle
      else if (depth >= Moho_depth && depth < LAB_depth && n_comp == mantle_L_id)
        return 1.;
      // Everything else, which is not represented by a compositional field.
      else
        return 0.;
    }


    template <int dim>
    void
    Litho1<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        Utilities::AsciiDataBoundary<dim>::declare_parameters(prm,
                                                          "$ASPECT_SOURCE_DIR/data/initial-composition/ascii-data/test/",
                                                          "box_2d_%s.%d.txt");
        prm.enter_subsection("LITHO1.0");
        {
          prm.declare_entry ("Upper crust fraction", "0.66",
                             Patterns::Double (0,1),
                             "A number that specifies the fraction of the Moho depth "
                             "that will be used to set the thickness of the upper crust "
                             "instead of reading its thickness from the ascii table. "
                             "For a fraction of 0, no compositional field is set for the "
                             "upper crust. "
                             "Unit: -.");
          prm.declare_entry ("Layer thicknesses", "30000.",
                             Patterns::List(Patterns::Double(0)),
                             "List of thicknesses for the bottom of the lithospheric layers,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value.  Units: $m$");
          prm.declare_entry ("Minimum LAB thickness", "80e3",
                             Patterns::Double (0),
                             "A number that specifies the minimum thickness of the lithosphere "
                             "regardless of the reading of its thickness from the ascii table. "
                             "Unit: m.");
          prm.declare_entry ("Merge LAB grids", "false",
                             Patterns::Bool (),
                             "Whether or not to merge the LAB thickness of two datasets "
                             "through smoothing with a hyperbolic tangent. "
                             "Unit: -.");
          prm.declare_entry ("LAB grid merge halfwidth", "1",
                             Patterns::Double (0),
                             "A number that specifies the halfwidth of the hyperbolic tangent "
                             "that is used to smooth the two LAB grid tables. "
                             "Unit: m or degrees.");
          prm.declare_entry ("LAB grid merge polygon", "",
                             Patterns::List(Patterns::Anything()),
                             "Set the polygon used to smooth the two LAB grids based on the "
                             "distance of the current point to the polygon."
                             "The polygon is a list of "
                             "points that represent horizontal coordinates (x,y) or (lon,lat). "
                             "The exact format for the point list describing a polygon is "
                             "\"x1,y1>x2,y2>x3,y3>x4,y4>x5,y5\". Note that the polygon is assumed to be closed. "
                             "The units of the coordinates are "
                             "dependent on the geometry model. In the box model they are in meters, in the "
                             "chunks they are in degrees.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    Litho1<dim>::parse_parameters (ParameterHandler &prm)
    {
      // Retrieve the indices of the fields that represent the lithospheric layers.
      AssertThrow(this->introspection().compositional_name_exists("upper"),ExcMessage("We need a compositional field called 'upper' representing the upper crust."));
      AssertThrow(this->introspection().compositional_name_exists("lower"),ExcMessage("We need a compositional field called 'lower' representing the lower crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_L"),ExcMessage("We need a compositional field called 'mantle_L' representing the lithospheric part of the mantle."));

      // For now, we assume a 3-layer system with an upper crust, lower crust and lithospheric mantle
      upper_crust_id = this->introspection().compositional_index_for_name("upper");
      lower_crust_id = this->introspection().compositional_index_for_name("lower");
      mantle_L_id = this->introspection().compositional_index_for_name("mantle_L");


      prm.enter_subsection("Initial composition model");
      {
        Utilities::AsciiDataBoundary<dim>::parse_parameters(prm);
        prm.enter_subsection("LITHO1.0");
        {
          upper_crust_fraction = prm.get_double ("Upper crust fraction");

//          // If there is no upper crust, the compositional field number
//          // of the lower crust is the same as that of the upper crust (i.e. they are the same field)
//          if (upper_crust_fraction == 0 && !this->introspection().compositional_name_exists("upper"))
//            lower_crust_id = 0;
//          else
//            lower_crust_id = 1;

          min_LAB_thickness = prm.get_double ("Minimum LAB thickness");
          merge_LAB_grids   = prm.get_bool ("Merge LAB grids");
          merge_LAB_grids_halfwidth = prm.get_double ("LAB grid merge halfwidth");

          // Split the string into point strings
          const std::vector<std::string> temp_points = Utilities::split_string_list(prm.get("LAB grid merge polygon"),'>');
          const unsigned int n_temp_points = temp_points.size();
          if (dim == 3)
            {
              AssertThrow(n_temp_points>=3, ExcMessage ("The number of polygon points should be equal to or larger than 3 in 3d."));
            }
          else
            {
              AssertThrow(n_temp_points==2, ExcMessage ("The number of polygon points should be equal to 2 in 2d."));
            }
          polygon_point_list.resize(n_temp_points);
          // Loop over the points of the polygon.
          for (unsigned int i_points = 0; i_points < n_temp_points; i_points++)
            {
              const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_points[i_points],','));
              Assert(temp_point.size() == dim-1,ExcMessage ("The given coordinates of point '" + temp_points[i_points] + "' are not correct. "
                                                            "It should only contain 1 (2d) or 2 (in 3d) parts: "
                                                            "the longitude/x (and latitude/y in 3d) coordinate (separated by a ',')."));

              // Add the point to the list of points for this segment
              polygon_point_list[i_points][0] = temp_point[0];
              polygon_point_list[i_points][1] = temp_point[dim-2];
            }
          if  (dim == 2)
            AssertThrow(polygon_point_list[0][0] < polygon_point_list[1][0], ExcMessage("The order of the x coordinates of the 2 points "
                "of each 2d polygon should be ascending. "));

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
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(Litho1,
                                              "litho1",
                                              "Implementation of a model in which the initial "
                                              "composition is derived from files containing data "
                                              "about the depth of the Moho and LAB "
                                              "in ascii format. "
                                              "Either one compositional field is used to represent "
                                              "the total crust up to Moho depth, "
                                              "or an upper crust composition is "
                                              "specified with the upper-crust-fraction parameter that"
                                              "sets an upper crustal field from 0 to "
                                              "upper-crust-fraction*Moho-depth. "
                                              "Note the required format of the "
                                              "input data: The first lines may contain any number of comments "
                                              "if they begin with '#', but one of these lines needs to "
                                              "contain the number of grid points in each dimension as "
                                              "for example '# POINTS: 3 3'. "
                                              "The order of the data columns "
                                              "has to be 'x', 'Moho depth [m]', 'LAB depth [m]', "
                                              "etc. in a 2d model and 'x', 'y', 'Moho depth [m]', "
                                              "'LAB depth [m]', etc. in a 3d model. "
                                              "Note that the data in the input "
                                              "files need to be sorted in a specific order: "
                                              "the first coordinate needs to ascend first, "
                                              "followed by the second in order to "
                                              "assign the correct data to the prescribed coordinates. "
                                              "If you use a spherical model, "
                                              "then the data will still be handled as Cartesian, "
                                              "however the assumed grid changes. 'x' will be replaced by "
                                              "by the azimuth angle and 'y' by the polar angle measured "
                                              "positive from the north pole. The grid will be assumed to be "
                                              "a latitude-longitude grid. Note that the order "
                                              "of spherical coordinates is 'phi', 'theta' "
                                              "and not 'theta', 'phi', since this allows "
                                              "for dimension independent expressions.")
  }
}
