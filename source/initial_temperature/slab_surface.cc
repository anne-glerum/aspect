/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


#include "aspect/initial_temperature/slab_surface.h"
#include <aspect/utilities.h>
#include <aspect/simulator_access.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/std_cxx1x/array.h>

namespace aspect
{
  namespace InitialTemperature
  {

    const double R_earth = 6371000.0;
//    const double R_model = 6380000.0;


//TODO: we don't need template parameter dim, we're only using this plugin for dim=3
    namespace internal
    {
      template <int dim>
      SlabGridLookup<dim>::SlabGridLookup (const unsigned int n_slab,
                                           const std::vector<unsigned int> n_hor,
                                           const std::vector<unsigned int> n_ver,
                                           const std::string &grid_file,
                                           const ConditionalOStream &pcout)
        :
        n_slabs(n_slab),
        n_hor_points(n_hor),
        n_ver_points(n_ver),
        grid_coord(n_slab,dealii::Table<2, Point<dim> > (0,0)),
        arc_length(n_slab,dealii::Table<2, double > (0,0))

      {
        pcout << std::endl << "   Opening slab grid file "
              << grid_file << "." << std::endl << std::endl;

        std::string temp;
        std::ifstream in_grid(grid_file.c_str(), std::ios::in);
        AssertThrow (in_grid,
                     ExcMessage (std::string("Couldn't open grid file <") + grid_file));

        for (unsigned int n = 0; n < n_slab; ++n)
          {
            pcout << "   Reading in slab grid " << n << " of total of " << grid_coord.size() << " slabs."
                  << std::endl << std::endl;

            const unsigned int n_grid_coord = dim * n_hor_points[n] * n_ver_points[n];

            std::vector<double> coords(0);


            for (unsigned int p = 0; p<n_grid_coord; p++)
              {
                double new_coord;
                if (!(in_grid >> new_coord))
                  {
                    AssertThrow (false, ExcMessage(std::string("Reading of coord with index ")
                                                   +
                                                   dealii::Utilities::int_to_string(p)
                                                   +
                                                   std::string(" failed. File corrupted? : ")
                                                   +
                                                   grid_file));
                  }

                coords.push_back(new_coord);
              }

            grid_coord[n].reinit(n_hor_points[n],n_ver_points[n]);
            arc_length[n].reinit(n_hor_points[n],n_ver_points[n]);

            unsigned int count = 0;
            // read in all coefficients
            for (unsigned int i = 0; i<n_hor_points[n]; i++)
              {
                for (unsigned int j = 0; j<n_ver_points[n]; j++)
                  {
                    for (unsigned int d = 0; d<dim; d++)
                      {
                        grid_coord[n][i][j][d] = coords[count];
                        ++count;
                      }
                    // Calculate the arc length for each coord point
                    if (j == 0)
                      {
                        arc_length[n][i][j] = 0.0;
                      }
                    else
                      {
                        arc_length[n][i][j] = arc_length[n][i][j-1] + grid_coord[n][i][j].distance(grid_coord[n][i][j-1]);
                      }
                  }
              }

            AssertThrow(count == n_grid_coord, ExcMessage("Number of read coordinates not as expected"));

            pcout << std::endl << "   Loaded "
                  << dealii::Utilities::int_to_string(count)
                  << " slab grid coordinates and calculated arc lengths"
                  << " of slab " << dealii::Utilities::int_to_string(n)
                  << std::endl << std::endl;

          }



      }


// Declare a function that returns the coord for a certain index
      template <int dim>
      Point<dim>
      SlabGridLookup<dim>::get_local_coord_point(const unsigned int slab_nr, const unsigned int i_hor, const unsigned int j_ver) const
      {
        Assert(slab_nr < n_slabs, ExcMessage("Slab number larger than the number of slabs. "));
        Assert(i_hor < n_hor_points[slab_nr] && j_ver < n_ver_points[slab_nr], ExcMessage("Grid point indices out of range: "
               + dealii::Utilities::int_to_string(i_hor)
               + ","
               + dealii::Utilities::int_to_string(j_ver)));

        return grid_coord[slab_nr][i_hor][j_ver];
      }

      template <int dim>
      double
      SlabGridLookup<dim>::get_arc_length(const unsigned int slab_nr, const unsigned int i_hor, const unsigned int j_ver) const
      {
        Assert(slab_nr < n_slabs, ExcMessage("Slab number larger than the number of slabs. "));
        Assert(i_hor < n_hor_points[slab_nr] && j_ver < n_ver_points[slab_nr], ExcMessage("Grid point indices too big: "
               + dealii::Utilities::int_to_string(i_hor)
               + ","
               + dealii::Utilities::int_to_string(j_ver)));
        return arc_length[slab_nr][i_hor][j_ver];

      }


    }

    template <int dim>
    void
    AsciiPip<dim>::initialize()
    {
      Adiabatic<dim>::initialize();

      Assert (dim == 3, ExcMessage ("This initial condition should be used in 3D."));

      if (n_slab_fields > 0)
        grid_lookup.reset(new internal::SlabGridLookup<dim>(n_slab_fields,n_hor_grid_points,n_ver_grid_points,datadirectory+slab_grid_file_name,this->get_pcout()));

    }



    template <int dim>
    double
    AsciiPip<dim>::
    initial_temperature (const Point<dim> &pos) const
    {
      //////////////////////////////////////////////////////////////////////////////////////////
      // There are five options for the temperature:                                          //
      // 1) point lies in slab: temperature will be described as in McKenzie 1970             //
      // 2) point lies in one of the other oceanic plates: plate cooling model                //
      // 3) point lies in one of the other continental plates: linear gradient                //
      // 4) point is neither slab nor plate, but mantle: adiabatic T                          //
      //    with/without T perturbation from tomography                                       //
      // 5) point lies in the air or ocean: T_top                                             //
      // We first create a background temperature field considering option 3 and 4 and then   //
      // check if we are in the plates/slabs and adjust the temperature accordingly.          //
      //////////////////////////////////////////////////////////////////////////////////////////

      // Depth of current point
      const double depth = this->get_geometry_model().depth(pos);
      // The temperature at the base of the adiabatic mantle
      Point<dim> spherical_coord;
      // TODO: adjust 500!
      // TODO: set lat and lon in adiab plugin because T_a is position dependent?
      spherical_coord[dim-1] = R_earth - max_lithosphere_depth;
      // TODO: get right point
      const Point<dim> cartesian_coord; // = polygon_lookup->spherical_to_cart(spherical_coord);
      T_a = this->get_adiabatic_conditions().temperature(cartesian_coord);

      double temperature = T_a;

      // Get the background temperature:
      // The adiabatic temperature at the bottom of the deepest plate
      // is uniformly set up to the depth of the deepest plate
      // Below that an adiabat with/without thermal anomalies from tomography
      // is prescribed.

//    else if (depth < Continental_Plate.get_depth_top()+Continental_Plate.get_thickness())
//      temperature = std::max(T_top,continental_plate_temperature(depth));
//      if (depth > max_lithosphere_depth)
//        temperature = ascii_model->initial_temperature(pos);

      // if deeper than slab_depth, return temperature right away
      if (depth > max_slab_depth)
        return std::max(T_top,std::min(temperature, T_bottom));

      temperature = std::max(T_top, slab_temperature(0, pos));

      return std::max(T_top,std::min(temperature, T_bottom));

    }

    template <int dim>
    double
    AsciiPip<dim>::
    oceanic_plate_temperature (const unsigned int field_nr,
                               const double depth,
                               const double plate_thickness,
                               const double plate_depth_top,
                               const double plate_age) const
    {

      // Thermal diffusivity
      // TODO: is this value representative?
      const double kappa = 1e-6;

      // The number of summations in the sum term
      const unsigned int n_sum = 80;
      double sum = 0;

      // Plate cooling model for a fixed age throughout the plate
      // (Schubert, Turcotte, Olson p. 139)

      // The thickness the plate would reach for large cooling ages
      const double old_L_thickness = 125000;

      for (unsigned int i=1; i<=n_sum; i++)
        {
          sum += (1.0/i) *
                 (exp((-kappa*i*i*numbers::PI*numbers::PI*plate_age)/(old_L_thickness*old_L_thickness)))*
                 (sin(i*numbers::PI*(depth-plate_depth_top)/old_L_thickness));
        }

      // The total temperature
      const double temp = T_top+(T_a-T_top)*( ((depth-plate_depth_top)/old_L_thickness) + (2.0/numbers::PI)*sum);

      // Some checks
      Assert (temp >= 0.0 && temp < 3000.0, ExcMessage("Oceanic plate temperature below 0 or above 3000 K."));

      return temp;
    }

    template <int dim>
    double
    AsciiPip<dim>::
    continental_plate_temperature (const double depth, const double plate_thickness, const double plate_depth_top) const
    {
      // Here we prescribe a linear profile
      const double temp = T_top + ((T_a - T_top) / plate_thickness ) * (depth - plate_depth_top);

      // Some checks
      Assert (temp >= 0.0 && temp < 3000.0, ExcMessage("Continental plate temperature below 0 or above 3000 K."));

      return temp;
    }


    template <int dim>
    double
    AsciiPip<dim>::
    slab_temperature (const unsigned int field_nr, const Point<dim> coord) const
    {
      Assert(slab_nr_per_volume[field_nr] != 10, ExcMessage("This field does not represent a slab. "));

      double temp = 0;

      // We use the adaptation of McKenzie 1970 by Pranger 2014, page 8.

      // The Reynolds number: reynolds = rho_0 * c_P * v * d / (2 * k) = v * d / (2 * kappa)
      // TODO: Are the reference values representative?
      const double reynolds = subduction_vel[slab_nr_per_volume[field_nr]] * slab_thickness[slab_nr_per_volume[field_nr]] / (2.0 * 1e-6);

      // The adiabatic term.
      // Because we set the temperature to T_a up to max_lithosphere_depth,
      // our adiabatic profile starts at max_lithosphere_depth,
      // so we subtract max_lithosphere depth from the depth of the current point.
      const double real_depth = this->get_geometry_model().depth(coord);
      const double depth = real_depth - max_lithosphere_depth;
      const double gravity = 9.81;
      const double exp_term = depth * 3e-5 * gravity / 1250.;

      // The number of summations in the summation term
      const unsigned int n_max = 80;
      double sum = 0;

      // The local coordinates along the local system parallel to
      // the down-dip direction (0) of the slab and the perpendicular
      // axis facing inwards (1).
      Tensor<1,2> local_coord = slab_length_and_depth(slab_nr_per_volume[field_nr],coord);

      // Sometimes the in-slab depth is slightly larger than the thickness
      // and sometimes the initial nearest_triangle_distance of 1e23
      // is not overwritten, so cap here.
      local_coord[1] = std::min(slab_thickness[slab_nr_per_volume[field_nr]],std::max(0.0,local_coord[1]));

      // The summation
      for (unsigned int n = 1; n <= n_max; ++n)
        {
          sum += std::pow(-1.0,n) *
                 (1.0 / (n * numbers::PI)) *
                 std::exp((reynolds - std::sqrt(reynolds * reynolds + (n * n * numbers::PI * numbers::PI))) * local_coord[0] / slab_thickness[slab_nr_per_volume[field_nr]] ) *
                 std::sin(n * numbers::PI * (1.0 - local_coord[1] / slab_thickness[slab_nr_per_volume[field_nr]]));
        }

      // The total temperature
      temp = std::exp(exp_term) * (T_a + 2.0 * (T_a - T_top) * sum);

      // If we are in the portion of slab that is not in contact with the mantle,
      // but with the overriding plate, we should consider prescribing just an
      // oceanic plate temperature distribution (so no heating from the slab's
      // surface by the mantle.
      const double oceanic_temp = oceanic_plate_temperature(field_nr,local_coord[1],slab_thickness[slab_nr_per_volume[field_nr]], 0.0, 200e6*year_in_seconds);
      // TODO: think about this
      // TODO: up to what depth should we do this? Overriding plate depth preferably.
      if (real_depth <= 120000.0 /*max_lithosphere_depth*/ && compensate_trench_temp)
        temp = std::min(temp, oceanic_temp);

      temp = std::max(temp, T_top);

      // Some checks
      Assert (temp >= 0.0 && temp < 3000.0, ExcMessage("Slab temperature below 0 or above 3000 K."));

      return temp;
    }


    template <int dim>
    Tensor<1,2>
    AsciiPip<dim>::
    slab_length_and_depth (const unsigned int slab_nr, const Point<dim> coord) const
    {
      Point<dim> pos = coord;

      // When looping over all slab grid points,
      // which grid point is closest to the queried point?
      double nearest_point_distance = 1e23;
      double distance;
      Tensor<1,2, unsigned int> nearest_index;

      // What 6 triangles (and what are their indices in terms of the
      // slab grid) surround the grid point closest to the queried point?
      std::vector<std::vector<std::vector<unsigned int> > > triangle_indices;

      // Which of the 6 triangles is closest to the projection of the
      // queried point to the surface of the slab (ie the slab grid)?
      bool in_triangle = false;
      bool nearest_point_in_triangle = false;
      double nearest_triangle_distance = 1e23;
      double normal_distance = 0.0;
      Point<dim> point_in_plane;
      Point<dim> nearest_point_in_plane;
      std::vector<Point<dim> > plane(3);
      std::vector<std::vector<unsigned int> > nearest_triangle_indices(3, std::vector<unsigned int>(2));

      // The arc lengths at the vertices of the
      // nearest triangle
      Tensor<1,3> triangle_arc_lengths;

      // The arc length in the projected point
      // from bi-linear interpolation of the arc
      // lengths in the vertices of the nearest triangle
      double arc_length = 0;

      // The result of this whole exercise:
      // the local coord arc length and in-slab depth
      Tensor<1,2> length_and_depth;

      // Calculate the nearest grid point to the queried point
      for (unsigned int i = 0; i < n_hor_grid_points[slab_nr]; ++i)
        {
          for (unsigned int j = n_ver_grid_points[slab_nr]; j-- > 0; )
            {
              const Point<dim> grid_point = grid_lookup->get_local_coord_point(slab_nr,i,j);
              distance = grid_point.distance(pos);

              if (distance < nearest_point_distance)
                {
                  nearest_point_distance = distance;
                  nearest_index[0] = i;
                  nearest_index[1] = j;
                }

            }
        }

      // Create the right dimensions: 6 triangles of each 3 points with 2 coordinates
      for (unsigned int i = 0; i < 6; ++i)
        {
          std::vector< std::vector<unsigned int> > new_vector(3, std::vector<unsigned int>(2));
          triangle_indices.push_back(new_vector);
        }


      // Now calculate nearest triangle around the point and interpolate the arclength of its 3 vertices on point
      // Assume grid is forward connected
      for (unsigned int i = 0; i < 6; ++i)
        {
          for (unsigned int j = 0; j < 3; ++j)
            {
              triangle_indices[i][j][0] = nearest_index[0];
              triangle_indices[i][j][1] = nearest_index[1];

            }
        }

      triangle_indices[0][1][0] += 1;
      triangle_indices[0][2][0] += 1;
      triangle_indices[0][2][1] += 1;

      triangle_indices[1][1][0] += 1;
      triangle_indices[1][1][1] += 1;
      triangle_indices[1][2][1] += 1;

      triangle_indices[2][1][1] += 1;
      triangle_indices[2][2][0] -= 1;

      triangle_indices[3][1][0] -= 1;
      triangle_indices[3][2][0] -= 1;
      triangle_indices[3][2][1] -= 1;

      triangle_indices[4][1][0] -= 1;
      triangle_indices[4][1][1] -= 1;
      triangle_indices[4][2][1] -= 1;

      triangle_indices[5][1][1] -= 1;
      triangle_indices[5][2][0] += 1;

      unsigned int count = 0;

      // Retrieve out of the 6 triangles surrounding the nearest slab
      // grid point the nearest triangle
      for (unsigned int i = 0; i < 6; ++i)
        {
          const std::vector<std::vector<unsigned int> > current_triangle_indices = triangle_indices[i];

          // These conditions seem to hold anyway, but..
          // additional triangles have been constructed possibly outside of the
          // grid, so we're checking for just the ones that are inside.
          // triangle_indices[0] is the nearest_grid_point, so no need to check
          if (current_triangle_indices[1][0] >= 0 && current_triangle_indices[1][0] < n_hor_grid_points[slab_nr] &&
              current_triangle_indices[1][1] >= 0 && current_triangle_indices[1][1] < n_ver_grid_points[slab_nr] &&
              current_triangle_indices[2][0] >= 0 && current_triangle_indices[2][0] < n_hor_grid_points[slab_nr] &&
              current_triangle_indices[2][1] >= 0 && current_triangle_indices[2][1] < n_ver_grid_points[slab_nr])
            {
              ++count;

              plane[0] = grid_lookup->get_local_coord_point(slab_nr,current_triangle_indices[0][0],current_triangle_indices[0][1]);
              plane[1] = grid_lookup->get_local_coord_point(slab_nr,current_triangle_indices[1][0],current_triangle_indices[1][1]);
              plane[2] = grid_lookup->get_local_coord_point(slab_nr,current_triangle_indices[2][0],current_triangle_indices[2][1]);

              point_to_plane(pos,plane,normal_distance,point_in_plane,in_triangle);

//            if (normal_distance < nearest_triangle_distance)
              if ((normal_distance < nearest_triangle_distance && in_triangle) || count == 1)
                {
                  nearest_triangle_distance = normal_distance;
                  nearest_triangle_indices = current_triangle_indices;
                  nearest_point_in_plane = point_in_plane;
                  nearest_point_in_triangle = in_triangle;
                }


            }
        }


      // Interpolate the arc lengths of each vertex of the nearest triangle
      // to the projected point
      plane[0] = grid_lookup->get_local_coord_point(slab_nr,nearest_triangle_indices[0][0],nearest_triangle_indices[0][1]);
      plane[1] = grid_lookup->get_local_coord_point(slab_nr,nearest_triangle_indices[1][0],nearest_triangle_indices[1][1]);
      plane[2] = grid_lookup->get_local_coord_point(slab_nr,nearest_triangle_indices[2][0],nearest_triangle_indices[2][1]);

      triangle_arc_lengths[0] = grid_lookup->get_arc_length(slab_nr,nearest_triangle_indices[0][0],nearest_triangle_indices[0][1]);
      triangle_arc_lengths[1] = grid_lookup->get_arc_length(slab_nr,nearest_triangle_indices[1][0],nearest_triangle_indices[1][1]);
      triangle_arc_lengths[2] = grid_lookup->get_arc_length(slab_nr,nearest_triangle_indices[2][0],nearest_triangle_indices[2][1]);

      arc_length = triangle_basis_function_interpolation(plane, triangle_arc_lengths, nearest_point_in_plane);

      // Yeey we made it
      length_and_depth[0] = std::max(arc_length, 0.0);
      length_and_depth[1] = nearest_triangle_distance;

      return length_and_depth;
    }

    template <int dim>
    void
    AsciiPip<dim>::point_to_plane(const Point<dim> coord, const std::vector<Point<dim> > plane,
                                  double &distance, Point<dim> &projection, bool &in_triangle) const
    {
      // "plane" is a vector of 3 points with each 3 coordinates
      Tensor<1,dim> vec_plane_1 = plane[1] - plane[0];
      Tensor<1,dim> vec_plane_2 = plane[2] - plane[0];
      Tensor<1,dim> vec_plane_position = coord - plane[0];
      Tensor<1,dim> normal;
      Tensor<1,dim> vec_4, vec_5;
      cross_product(normal,vec_plane_1,vec_plane_2);
      const double epsilon = 1e-9;
      double cos_angle(0);
      // The coordinates of the projected point within the triangle
      Tensor<1,2> local_coord;

      if (normal.norm() < epsilon)
        {
          cross_product(vec_4, vec_plane_2, vec_plane_position);
          cross_product(normal, vec_plane_2, vec_4);
          if (normal.norm() < epsilon)
            {
              cross_product(vec_4, vec_plane_1, vec_plane_position);
              cross_product(normal, vec_plane_1, vec_4);
            }
        }

      const double norm_vec_plane_position = vec_plane_position.norm();
      const double norm_normal = normal.norm();
      if (norm_vec_plane_position > epsilon && norm_normal > epsilon)
        {
          cos_angle = (vec_plane_position/norm_vec_plane_position) * (normal/norm_normal);
          distance  = std::abs(norm_vec_plane_position * cos_angle);
          projection = coord - norm_vec_plane_position * cos_angle * normal/norm_normal;

          triangle_coord(projection, plane, local_coord);

          if (local_coord[0] >= 0.0 && local_coord[1] >= 0.0 && local_coord[0]+local_coord[1] <= 1.0)
            {
              in_triangle = true;
            }
        }
      else
        {
          distance = 0.0;
          projection = plane[0];
          in_triangle = false;
        }

    }

    template <int dim>
    void
    AsciiPip<dim>::triangle_coord(const Point<dim> coord, const std::vector<Point<dim> > triangle, Tensor<1,2> &triangle_coord) const
    {

      // The transformation matrix made up of the axes of the new coordinate system
      Tensor<2,dim> transf_matrix;

      // The inverse of the transformation matrix
      Tensor<2,dim> inverse;

      // The determinant of the transformation matrix
      double determ(0);

      // The point in the local triangle coordinates
      Tensor<1,dim> new_point;

      Tensor<1,dim> point_0 = coord - triangle[0];

      const double epsilon = 1e-9;

      // The new axes in the old coordinate system
      Tensor<1,dim> new_x_axis = triangle[1] - triangle[0];
      Tensor<1,dim> new_y_axis = triangle[2] - triangle[0];
      Tensor<1,dim> new_z_axis;
      cross_product(new_z_axis,new_x_axis,new_y_axis);

      // A helper vector for finding the perpendicular z-axis
      Point<dim> temp_vec;

      if (new_z_axis.norm() < epsilon)
        {
          cross_product(temp_vec, new_y_axis, point_0);
          cross_product(new_z_axis, new_y_axis, temp_vec);
          if (new_z_axis.norm() < epsilon)
            {
              cross_product(temp_vec, new_x_axis, point_0);
              cross_product(new_z_axis, new_x_axis, temp_vec);
            }
        }

      // Fill the transformation matrix columns with the axes
      // of the triangle AB, AC and the computed z-axis.
      for (unsigned int d = 0; d<dim; d++)
        {
          transf_matrix[d][0] = new_x_axis[d];
          transf_matrix[d][1] = new_y_axis[d];
          transf_matrix[d][2] = new_z_axis[d];
        }

      determ = determinant(transf_matrix);

      // If the determinant is zero, we're not going to compute the inverse
      if (determ < epsilon)
        {
          triangle_coord[0] = -0.1;
          triangle_coord[1] = -0.1;
        }
      // Compute the inverse and the coordinates of the point in the
      // triangle coordinate system
      else
        {
          Tensor<2,dim> inverse = invert(transf_matrix);
          new_point = inverse * point_0;

          // Drop the third coordinate
          triangle_coord[0] = new_point[0];
          triangle_coord[1] = new_point[1];
        }

    }

    template <int dim>
    double
    AsciiPip<dim>::triangle_basis_function_interpolation( const std::vector<Point<dim> > triangle, const Tensor<1,3> values, const Point<dim>  point) const
    {
      double interpolated_value = 0;
      Tensor<1,2> local_coord;

      triangle_coord(point, triangle, local_coord);

      interpolated_value = values[0] * (1.0 - local_coord[0] - local_coord[1]) +
                           values[1] * local_coord[0] +
                           values[2] * local_coord[1];

      return interpolated_value;
    }


    template <int dim>
    void
    AsciiPip<dim>::declare_parameters (ParameterHandler &prm)
    {
      Adiabatic<dim>::declare_parameters(prm);

      prm.enter_subsection ("Initial temperature model");
      {
        prm.enter_subsection("Pip");
        {
          prm.declare_entry("Base model","adiabatic profile with ascii perturbations",
                            Patterns::Selection("adiabatic profile with ascii perturbations"),
                            "The name of a temperature model that will be used "
                            "for the mantle temperature. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Initial conditions models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.declare_entry("Data directory", "$ASPECT_SOURCE_DIR/data/initial-temperature/pip/",
                            Patterns::DirectoryName (),
                            "The path to the model data. ");

          prm.declare_entry ("Slab grid file name", "grid_coord.dat",
                             Patterns::Anything(),
                             "The file name of the slab grid points "
                             "of a 3D varying slab surface.");

          prm.declare_entry ("Number of slabs", "0",
                             Patterns::Integer (0),
                             "This parameter specifies the number of slabs "
                             "for which there are grids in the slab grid file.");

          prm.declare_entry ("Number of horizontal grid points per slab", "",
                             Patterns::List (Patterns::Integer (0)),
                             "This parameter specifies the number of horizontal grid points "
                             "in the regular grid along the surface of the slab per slab.");

          prm.declare_entry ("Number of vertical grid points per slab", "",
                             Patterns::List (Patterns::Integer (0)),
                             "This parameter specifies the number of vertical grid points "
                             "in the regular grid along the surface of the slab per slab.");

          prm.declare_entry ("Slab max depth", "700e3",
                             Patterns::List (Patterns::Double (0)),
                             "This parameter specifies the maximum depth of all slabs. "
                             "If temperature is requested for a point of deeper depth, the mantle temperature is returned directly.");

          prm.declare_entry ("Slab thickness per slab", "60000",
                             Patterns::List (Patterns::Double (0)),
                             "This parameter specifies the constant thickness of the "
                             "subducting slab as needed for the initial T calculation per slab.");

          prm.declare_entry ("Slab subduction velocity per slab", "3.17e-10",
                             Patterns::List (Patterns::Double (0)),
                             "This parameter specifies the assumed historically constant "
                             "subduction velocity as needed for the initial T calculation per slab.");

          prm.declare_entry ("Slab number per field", "",
                             Patterns::List (Patterns::Integer (0)),
                             "This parameter lists the number of the slab as in the order "
                             "of the slab grid file, starting at 0, if a field represents a slab."
                             "Otherwise set number to 10.");

          prm.declare_entry("Adjust trench temperature", "true",
                            Patterns::Bool(),
                            "This parameter specifies whether to use a normal cooling half-space model for "
                            "the temperature in the slab in contact with overriding plate (true) or to use "
                            "the McKenzie 1974 formulation in the whole slab (false).");

        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    AsciiPip<dim>::parse_parameters (ParameterHandler &prm)
    {
      Adiabatic<dim>::parse_parameters(prm);

      prm.enter_subsection ("Initial temperature model");
      {
        prm.enter_subsection("Slab surface");
        {
          // We use the adiabatic plugin for the background temperature
          Adiabatic->parse_parameters(prm);

          // Where we get our slab data file(s)
          datadirectory           = prm.get ("Data directory");
          {
            const std::string      subst_text = "$ASPECT_SOURCE_DIR";
            std::string::size_type position;
            while (position = datadirectory.find (subst_text),  position!=std::string::npos)
              datadirectory.replace (datadirectory.begin()+position,
                                     datadirectory.begin()+position+subst_text.size(),
                                     subst_text);
          }
          slab_grid_file_name = prm.get ("Slab grid file name");

          // Slab information
          n_slab_fields       = prm.get_integer ("Number of slabs");
          max_slab_depth      = prm.get_double("Slab max depth");

          const std::vector<int> n_hor = dealii::Utilities::string_to_int
                                         (dealii::Utilities::split_string_list(prm.get("Number of horizontal grid points per slab")));
          AssertThrow (n_hor.size() == n_slab_fields, ExcMessage("The number of slabs for which horizontal grid points are specified "
                                                                 "does not correspond to the number of slabs. "));
          n_hor_grid_points = std::vector<unsigned int> (n_hor.begin(),
                                                         n_hor.end());

          const std::vector<int> n_ver = dealii::Utilities::string_to_int
                                         (dealii::Utilities::split_string_list(prm.get("Number of vertical grid points per slab")));
          AssertThrow (n_ver.size() == n_slab_fields, ExcMessage("The number of slabs for which vertical grid points are specified "
                                                                 "does not correspond to the number of slabs. "));
          n_ver_grid_points = std::vector<unsigned int> (n_ver.begin(),
                                                         n_ver.end());

          // TODO: get from plate type
          slab_thickness = dealii::Utilities::string_to_double
                           (dealii::Utilities::split_string_list(prm.get("Slab thickness per slab")));
          AssertThrow (slab_thickness.size() == n_slab_fields, ExcMessage("The number of slabs for which a thickness is specified "
                                                                          "does not correspond to the number of slabs. "));

          subduction_vel = dealii::Utilities::string_to_double
                           (dealii::Utilities::split_string_list(prm.get("Slab subduction velocity per slab")));
          AssertThrow (subduction_vel.size() == n_slab_fields, ExcMessage("The number of slabs for which vertical grid points are specified "
                                                                          "does not correspond to the number of slabs. "));


          compensate_trench_temp = prm.get_bool("Adjust trench temperature");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();



      // TODO: read these in from somewhere!
//    T_top = 285.0;
      T_a   = 1650.0;
//    T_bottom = 3590.0;
      prm.enter_subsection("Boundary temperature model");
      {
        prm.enter_subsection("Spherical constant");
        {
          T_bottom = prm.get_double ("Inner temperature");
          T_top = prm.get_double ("Outer temperature");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();




    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(AsciiPip,
                                              "ascii pip",
                                              "An initial temperature field in which the temperature "
                                              "is according to the plate cooling model unless the queried"
                                              "point lies in the slab, then McKenzie 1970 is followed.")
  }
}
