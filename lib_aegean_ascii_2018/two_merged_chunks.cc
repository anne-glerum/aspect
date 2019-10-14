/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include "two_merged_chunks.h"
#include <aspect/geometry_model/initial_topography_model/zero_topography.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold_lib.h>


namespace aspect
{
  namespace GeometryModel
  {
    template <int dim>
    Point<dim>
    TwoMergedChunks<dim>::ChunkGeometry::
    push_forward(const Point<dim> &input_vertex) const
    {
      Point<dim> output_vertex;
      switch (dim)
        {
          case 2:
          {
            output_vertex[0] = input_vertex[0]*std::cos(input_vertex[1]); // x=rcosphi
            output_vertex[1] = input_vertex[0]*std::sin(input_vertex[1]); // z=rsinphi
            break;
          }
          case 3:
          {
            output_vertex[0] = input_vertex[0]*std::cos(input_vertex[2])*std::cos(input_vertex[1]); // x=rsinthetacosphi
            output_vertex[1] = input_vertex[0]*std::cos(input_vertex[2])*std::sin(input_vertex[1]); // y=rsinthetasinphi
            output_vertex[2] = input_vertex[0]*std::sin(input_vertex[2]); // z=rcostheta
            break;
          }
          default:
            Assert (false, ExcNotImplemented ());
        }
      return output_vertex;
    }

    template <int dim>
    Point<dim>
    TwoMergedChunks<dim>::ChunkGeometry::
    pull_back(const Point<dim> &v) const
    {
      Point<dim> output_vertex;
      switch (dim)
        {
          case 2:
          {
            output_vertex[1] = std::atan2(v[1], v[0]);
            output_vertex[0] = v.norm();
            break;
          }
          case 3:
          {
            const double radius=v.norm();
            output_vertex[0] = radius;
            output_vertex[1] = std::atan2(v[1], v[0]);
            output_vertex[2] = std::asin(v[2]/radius);
            break;
          }
          default:
            Assert (false, ExcNotImplemented ());
        }
      return output_vertex;
    }

    template <int dim>
    void
    TwoMergedChunks<dim>::
    set_boundary_indicators (parallel::distributed::Triangulation<dim> &triangulation) const
    {
      // iterate over all active cells and (re)set the boundary indicators
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {

          // first set the default boundary indicators
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              cell->face(f)->set_all_boundary_ids (f);

          if (cell->face(3)->at_boundary())
            // set the upper part of the eastern boundary to indicator 2*dim+1
            if ((cell->vertex(GeometryInfo<dim-1>::vertices_per_cell-1).norm() + cell->vertex(0).norm()) / 2.0 > point3[0])
              cell->face(3)->set_all_boundary_ids (2*dim+1);

          if (cell->face(2)->at_boundary())
            // set the upper part of the western boundary to indicator 2*dim
            if ((cell->vertex(GeometryInfo<dim-1>::vertices_per_cell-1).norm() + cell->vertex(0).norm()) / 2.0 > point3[0])
              cell->face(2)->set_all_boundary_ids (2*dim);

          if (dim==3)
            {
              // set the upper part of the southern boundary to indicator 2*dim+2
              if (cell->face(4)->at_boundary())
                if ((cell->vertex(GeometryInfo<dim-1>::vertices_per_cell-1).norm() + cell->vertex(0).norm()) / 2.0 > point3[0])
                  cell->face(4)->set_all_boundary_ids (2*dim+2);
              // set the upper part of the northern boundary to indicator 2*dim+3
              if (cell->face(5)->at_boundary())
                if ((cell->vertex((GeometryInfo<dim-1>::vertices_per_cell-1)/2).norm()  + cell->vertex(0).norm()) / 2.0 > point3[0])
                  cell->face(5)->set_all_boundary_ids (2*dim+3);
            }

            // reset the curved boundary ids
            if (cell->face(0)->at_boundary())
              cell->face(0)->set_all_boundary_ids (0);
            if (cell->face(1)->at_boundary())
              cell->face(1)->set_all_boundary_ids (1);

        }
    }

    template <int dim>
    void
    TwoMergedChunks<dim>::
    create_coarse_mesh (parallel::distributed::Triangulation<dim> &total_coarse_grid) const
    {
      std::vector<unsigned int> lower_rep_vec(lower_repetitions, lower_repetitions+dim);
      std::vector<unsigned int> upper_rep_vec(upper_repetitions, upper_repetitions+dim);

      // the two triangulations that will be merged
      Triangulation<dim> lower_coarse_grid;
      Triangulation<dim> upper_coarse_grid;

      // create lower box
      GridGenerator::subdivided_hyper_rectangle (lower_coarse_grid,
                                                 lower_rep_vec,
                                                 point1,
                                                 point4,
                                                 false);


      // create upper box
      GridGenerator::subdivided_hyper_rectangle (upper_coarse_grid,
                                                 upper_rep_vec,
                                                 point3,
                                                 point2,
                                                 false);

      // merge the lower and upper mesh into one total_coarse_grid.
      // now we have at least two cells
      GridGenerator::merge_triangulations(lower_coarse_grid,
                                          upper_coarse_grid,
                                          total_coarse_grid);

      // Transform box into spherical chunk
      GridTools::transform (std::bind(&ChunkGeometry::push_forward,
                                            std::cref(manifold),
                                            std::placeholders::_1),
                            total_coarse_grid);

      // Deal with a curved mesh
      // Attach the real manifold to slot 15. we won't use it
      // during regular operation, but we set manifold_ids for all
      // cells, faces and edges immediately before refinement and
      // clear it again afterwards
      total_coarse_grid.set_manifold (15, manifold);

      total_coarse_grid.signals.pre_refinement.connect (std::bind (&set_manifold_ids,
                                                                         std::ref(total_coarse_grid)));
      total_coarse_grid.signals.post_refinement.connect (std::bind (&clear_manifold_ids,
                                                                          std::ref(total_coarse_grid)));

      // set the boundary indicators
      set_boundary_indicators(total_coarse_grid);
      // make sure the right boundary indicators are set after refinement
      // through the function set_boundary_indicators above
      total_coarse_grid.signals.post_refinement.connect
      (std::bind (&TwoMergedChunks<dim>::set_boundary_indicators,
                        std::cref(*this),
                        std::ref(total_coarse_grid)));
    }

    template <int dim>
    std::set<types::boundary_id>
    TwoMergedChunks<dim>::
    get_used_boundary_indicators () const
    {
      // boundary indicators are zero through 2*dim+2*(dim-1)-1
      std::set<types::boundary_id> s;
      for (unsigned int i=0; i<2*dim+2*(dim-1); ++i)
        s.insert (i);
      return s;
    }



    template <int dim>
    double
    TwoMergedChunks<dim>::height_above_reference_surface(const Point<dim> &position) const
    {   
      return position.norm()-point2[0];
    } 


    template <int dim>
    std::array<double,dim>
    TwoMergedChunks<dim>::cartesian_to_natural_coordinates(const Point<dim> &position_point) const
    {
      // the chunk manifold has a order of radius, longitude, latitude.
      // This is exactly what we need.
      const Point<dim> transformed_point = manifold.pull_back(position_point);
      std::array<double,dim> position_array;
      for (unsigned int i = 0; i < dim; i++)
        position_array[i] = transformed_point(i);

      return position_array;
    }



    template <int dim>
    aspect::Utilities::Coordinates::CoordinateSystem
    TwoMergedChunks<dim>::natural_coordinate_system() const
    {
      return aspect::Utilities::Coordinates::CoordinateSystem::spherical;
    }



    template <int dim>
    Point<dim>
    TwoMergedChunks<dim>::natural_to_cartesian_coordinates(const std::array<double,dim> &position_tensor) const
    {
      Point<dim> position_point;
      for (unsigned int i = 0; i < dim; i++)
        position_point[i] = position_tensor[i];
      const Point<dim> transformed_point = manifold.push_forward(position_point);

      return transformed_point;
    }


    template <int dim>
    std::map<std::string,types::boundary_id>
    TwoMergedChunks<dim>::
    get_symbolic_boundary_names_map () const
    {
      switch (dim)
        {
          case 2:
          {
            static const std::pair<std::string,types::boundary_id> mapping[]
              = { std::pair<std::string,types::boundary_id>("bottom",  0),
                  std::pair<std::string,types::boundary_id>("top",  1),
                  std::pair<std::string,types::boundary_id>("lowereast",   2),
                  std::pair<std::string,types::boundary_id>("lowerwest",   3),
                  std::pair<std::string,types::boundary_id>("uppereast",   4),
                  std::pair<std::string,types::boundary_id>("upperwest",   5)
                };

            return std::map<std::string,types::boundary_id> (&mapping[0],
                                                             &mapping[sizeof(mapping)/sizeof(mapping[0])]);
          }

          case 3:
          {
            static const std::pair<std::string,types::boundary_id> mapping[]
              = { std::pair<std::string,types::boundary_id>("bottom",  0),
                  std::pair<std::string,types::boundary_id>("top",  1),
                  std::pair<std::string,types::boundary_id>("lowerwest",   2),
                  std::pair<std::string,types::boundary_id>("lowereast",   3),
                  std::pair<std::string,types::boundary_id>("lowersouth",  4),
                  std::pair<std::string,types::boundary_id>("lowernorth",  5),
                  std::pair<std::string,types::boundary_id>("upperwest",   6),
                  std::pair<std::string,types::boundary_id>("uppereast",   7),
                  std::pair<std::string,types::boundary_id>("uppersouth",  8),
                  std::pair<std::string,types::boundary_id>("uppernorth",  9)

                };

            return std::map<std::string,types::boundary_id> (&mapping[0],
                                                             &mapping[sizeof(mapping)/sizeof(mapping[0])]);
          }
        }

      Assert (false, ExcNotImplemented());
      return std::map<std::string,types::boundary_id>();
    }


    template <int dim>
    double
    TwoMergedChunks<dim>::
    length_scale () const
    {
      // As described in the first ASPECT paper, a length scale of
      // 10km = 1e4m works well for the pressure scaling for earth
      // sized spherical shells. use a length scale that
      // yields this value for the R0,R1 corresponding to earth
      // but otherwise scales like (R1-R0)
      return 1e4 * maximal_depth() / (6336000.-3481000.);
    }


    template <int dim>
    double
    TwoMergedChunks<dim>::depth(const Point<dim> &position) const
    {
      return std::min (std::max (point2[0]-position.norm(), 0.), maximal_depth());
    }


    template <int dim>
    Point<dim>
    TwoMergedChunks<dim>::representative_point(const double depth) const
    {
      Assert (depth >= 0,
              ExcMessage ("Given depth must be positive or zero."));
      Assert (depth <= maximal_depth(),
              ExcMessage ("Given depth must be less than or equal to the maximal depth of this geometry."));

      // Choose a point at the mean longitude (and latitude)
      Point<dim> p = 0.5*(point2+point1);
      // at a depth beneath the top surface
      p[0] = point2[0]-depth;

      // Adapt latitude so point lies in representative continental lithosphere
      if (dim == 3)
        p[2] = numbers::PI / 180.0 * -15.0;

      // Now convert to Cartesian coordinates
      return manifold.push_forward(p);
    }

    template <int dim>
    double
    TwoMergedChunks<dim>::west_longitude () const
    {
      return point1[1];
    }


    template <int dim>
    double
    TwoMergedChunks<dim>::east_longitude () const
    {
      return point2[1];
    }

    template <int dim>
    double
    TwoMergedChunks<dim>::longitude_range () const
    {
      return point2[1] - point1[1];
    }

    template <int dim>
    double
    TwoMergedChunks<dim>::south_latitude () const
    {
      if (dim == 3)
        return point1[2];
      else
        return 0;
    }


    template <int dim>
    double
    TwoMergedChunks<dim>::north_latitude () const
    {
      if (dim==3)
        return point2[2];
      else
        return 0;
    }


    template <int dim>
    double
    TwoMergedChunks<dim>::latitude_range () const
    {
      if (dim==3)
        return point2[2] - point1[2];
      else
        return 0;
    }


    template <int dim>
    double
    TwoMergedChunks<dim>::maximal_depth() const
    {
      return point2[0]-point1[0];
    }

    template <int dim>
    double
    TwoMergedChunks<dim>::inner_radius () const
    {
      return point1[0];
    }

    template <int dim>
    double
    TwoMergedChunks<dim>::outer_radius () const
    {
      return point2[0];
    }

    template <int dim>
    bool
    TwoMergedChunks<dim>::has_curved_elements() const
    {
      return true;
    }

    template <int dim>
    bool
    TwoMergedChunks<dim>::point_is_in_domain(const Point<dim> &point) const
    {
      AssertThrow(this->get_free_surface_boundary_indicators().size() == 0 ||
                  this->get_timestep_number() == 0,
                  ExcMessage("After displacement of the free surface, this function can no longer be used to determine whether a point lies in the domain or not."));

      AssertThrow(dynamic_cast<const InitialTopographyModel::ZeroTopography<dim>*>(&this->get_initial_topography_model()) != 0,
                  ExcMessage("After adding topography, this function can no longer be used to determine whether a point lies in the domain or not."));

      const Point<dim> spherical_point = manifold.pull_back(point);

      for (unsigned int d = 0; d < dim; d++)
        if (spherical_point[d] > point2[d]+std::numeric_limits<double>::epsilon()*std::abs(point2[d]) ||
            spherical_point[d] < point1[d]-std::numeric_limits<double>::epsilon()*std::abs(point2[d]))
          return false;

      return true;
    }

    template <int dim>
    void
    TwoMergedChunks<dim>::set_manifold_ids (Triangulation<dim> &triangulation)
    {
      for (typename Triangulation<dim>::active_cell_iterator cell =
             triangulation.begin_active(); cell != triangulation.end(); ++cell)
        cell->set_all_manifold_ids (15);
    }

    template <int dim>
    void
    TwoMergedChunks<dim>::clear_manifold_ids (Triangulation<dim> &triangulation)
    {
      for (typename Triangulation<dim>::active_cell_iterator cell =
             triangulation.begin_active(); cell != triangulation.end(); ++cell)
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->at_boundary(f))
            cell->face(f)->set_all_manifold_ids (numbers::invalid_manifold_id);
      //cell->set_all_manifold_ids (numbers::invalid_manifold_id);
    }

    template <int dim>
    void
    TwoMergedChunks<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Two merged chunks");
        {
          prm.declare_entry ("Chunk inner radius", "0",
                             Patterns::Double (0),
                             "Radius at the bottom surface of the chunk. Units: m.");
          prm.declare_entry ("Chunk outer radius", "1",
                             Patterns::Double (0),
                             "Radius at the top surface of the chunk. Units: m.");
          prm.declare_entry ("Chunk grid merge radius", "1",
                             Patterns::Double (0),
                             "Radius at the top surface of the lower chunk, "
                             "where the two grids merge. Units: m.");

          prm.declare_entry ("Chunk minimum longitude", "0",
                             Patterns::Double (-180, 360), // enables crossing of either hemisphere
                             "Minimum longitude of the chunk. Units: degrees.");
          prm.declare_entry ("Chunk maximum longitude", "1",
                             Patterns::Double (-180, 360), // enables crossing of either hemisphere
                             "Maximum longitude of the chunk. Units: degrees.");

          prm.declare_entry ("Chunk minimum latitude", "0",
                             Patterns::Double (-90, 90),
                             "Minimum latitude of the chunk. This value is ignored "
                             "if the simulation is in 2d. Units: degrees.");
          prm.declare_entry ("Chunk maximum latitude", "1",
                             Patterns::Double (-90, 90),
                             "Maximum latitude of the chunk. This value is ignored "
                             "if the simulation is in 2d. Units: degrees.");

          prm.declare_entry ("Outer radius repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in radius.");
          prm.declare_entry ("Grid merge radius repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in radius.");
          prm.declare_entry ("Longitude repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in longitude for the lower chunk.");
          prm.declare_entry ("Latitude repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in latitude for the lower chunk. This value is ignored "
                             "if the simulation is in 2d");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    TwoMergedChunks<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry model");
      {
        prm.enter_subsection("Two merged chunks");
        {

          const double degtorad = dealii::numbers::PI/180;

          Assert (dim >= 2, ExcInternalError());
          Assert (dim <= 3, ExcInternalError());

          if (dim >= 2)
            {
              point1[0] = prm.get_double ("Chunk inner radius");
              point2[0] = prm.get_double ("Chunk outer radius");
              point3[0] = prm.get_double ("Chunk grid merge radius");
              point4[0] = point3[0];
              lower_repetitions[0] = prm.get_integer ("Grid merge radius repetitions");
              upper_repetitions[0] = prm.get_integer ("Outer radius repetitions");
              point1[1] = prm.get_double ("Chunk minimum longitude") * degtorad;
              point2[1] = prm.get_double ("Chunk maximum longitude") * degtorad;
              point3[1] = point1[1];
              point4[1] = point2[1];
              lower_repetitions[1] = prm.get_integer ("Longitude repetitions");
              upper_repetitions[1] = lower_repetitions[1];

              AssertThrow (point1[0] < point2[0],
                           ExcMessage ("Inner radius must be less than outer radius."));
              AssertThrow (point3[0] < point2[0],
                           ExcMessage ("Grid merge radius must be less than outer radius."));
              AssertThrow (point1[1] < point2[1],
                           ExcMessage ("Minimum longitude must be less than maximum longitude."));
              AssertThrow (point2[1] - point1[1] < 2.*numbers::PI,
                           ExcMessage ("Maximum - minimum longitude should be less than 360 degrees."));
            }

          if (dim == 3)
            {
              point1[2] = prm.get_double ("Chunk minimum latitude") * degtorad;
              point2[2] = prm.get_double ("Chunk maximum latitude") * degtorad;
              point3[2] = point1[2];
              point4[2] = point2[2];
              lower_repetitions[2] = prm.get_integer ("Latitude repetitions");
              upper_repetitions[2] = lower_repetitions[2];

              AssertThrow (point1[2] < point2[2],
                           ExcMessage ("Minimum latitude must be less than maximum latitude."));
            }

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
  namespace GeometryModel
  {
    ASPECT_REGISTER_GEOMETRY_MODEL(TwoMergedChunks,
                                   "two merged chunks",
                                   "A geometry which can be described as a chunk of a spherical shell, "
                                   "bounded by lines of longitude, latitude and radius. "
                                   "The radial boundaries have two boundary indicators, so the user "
                                   "can prescribe different boundary conditions on these boundaries. "
                                   "The minimum and maximum longitude, (latitude) and depth of the chunk "
                                   "are set in the parameter file. The chunk geometry labels its "
                                   "2*dim+2*(dim-1) sides as follows: ``lower west'' and ``lower east'': "
                                   "minimum and maximum longitude of the lower part of the east and west "
                                   "radial boundaries, ``upper west and upper east'': "
                                   "minimum and maximum longitude of the upper part of the east and west "
                                   "radial boundaries, ``lower south'' and ``lower north'': "
                                   "minimum and maximum latitude of the lower part of the south and north "
                                   "radial boundaries, ``upper south'' and ``upper north'': "
                                   "minimum and maximum latitude of the upper part of the south and north "
                                   "radial boundaries, "
                                   "``inner'' and ``outer'': minimum and maximum radii. "
                                   "Names in the parameter files are as follows: "
                                   "Chunk (minimum || maximum) (longitude || latitude): "
                                   "edges of geographical quadrangle (in degrees)"
                                   "Chunk (inner || outer) radius: Radii at bottom and top of chunk"
                                   "Chunk grid merge radius: Radius where the change in boundary indicator "
                                   "should occur"
                                   "(Longitude || Latitude) repetitions: "
                                   "number of cells in each coordinate direction."
                                   "(Grid merge radius || Outer radius) repetitions: number of cells in the "
                                   "radial direction in the upper and lower part of the domain.")
  }
}

