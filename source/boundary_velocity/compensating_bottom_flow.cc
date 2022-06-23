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


#include <aspect/global.h>
#include <aspect/utilities.h>

#include <aspect/boundary_velocity/interface.h>
#include <aspect/boundary_velocity/compensating_bottom_flow.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/ellipsoidal_chunk.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace BoundaryVelocity
  {

    template <int dim>
    CompensatingBottomFlow<dim>::CompensatingBottomFlow ()
    {}


    template <int dim>
    void
    CompensatingBottomFlow<dim>::initialize ()
    {
      // AssertThrow (/*((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model())) != 0)
      //              || */((dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0)
      //              || ((dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model())) != 0),
      //              ExcMessage ("This Euler pole plugin can only be used when using "
      //                  "a (ellipsoidal) chunk geometry."));

      double dlon = 0;
      double min_lat = 0, max_lat = 0;

      // set inner and outer radius
      if (const GeometryModel::SphericalShell<dim> *gm = dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()))
        {
          inner_radius = gm->inner_radius();
          outer_radius = gm->outer_radius();
          dlon = gm->opening_angle();
          // in 3D, geometry is either a spherical shell or a quarter shell
          if (dlon == 360.)
            {
              AssertThrow (false, ExcMessage("The Compensating Bottom Flow velocity boundary conditions plugin requires lateral boundaries."));
              max_lat = numbers::PI;
            }
          else
            max_lat = 0.5 * numbers::PI;
          dlon *= numbers::PI / 180.;
        }
      else if (const GeometryModel::Chunk<dim> *gm = dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model()))
        {
          inner_radius = gm->inner_radius();
          outer_radius = gm->outer_radius();
          dlon = gm->longitude_range();
          // colat
          min_lat = 0.5 * numbers::PI - gm->north_latitude();
          max_lat = 0.5 * numbers::PI - gm->south_latitude();
        }
      else if (const GeometryModel::EllipsoidalChunk<dim> *gm = dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model()))
        {
          outer_radius = gm->get_semi_major_axis_a();
          inner_radius = outer_radius - gm->maximal_depth();

          // Assume chunk outlines are lat/lon parallel
          std::vector<Point<2>> corners = gm->get_corners();
          AssertThrow(corners[0][0]==corners[3][0] && corners[0][1]==corners[1][1],
                      ExcMessage("This boundary velocity plugin cannot be used when the domain boundaries are not parallel to the lat/lon grid."));

          // colat
          min_lat = (90. - corners[0][1]) * numbers::PI / 180.;
          max_lat = (90. - corners[2][1]) * numbers::PI / 180.;

          const double max_lon = corners[0][0];
          const double min_lon = corners[1][0];
          dlon = (max_lon - min_lon) * numbers::PI / 180.;
        }
      else if (const GeometryModel::Box<dim> *gm = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()))
        {
          const Point<dim> extents = gm->get_extents();
          const Point<dim> origin = gm->get_origin();
          // TODO rename parameters to also make sense in cartesian domain
          inner_radius = origin[dim-1];
          outer_radius = extents[dim-1] + origin[dim-1];
          dlon = extents[0];
          if (dim == 3)
            {
              min_lat = origin[dim-2];
              max_lat = origin[dim-2] + extents[dim-2];
            }
        }
      else
        AssertThrow(false, ExcMessage("The Compensating Bottom Flow boundary velocity plugin does not work for this geometry model."));



      // Compute the area of the bottom boundary
      // as the integral over longitude interval dlon and latitude interval dlat of R0*R0*sin(lat)
      // TODO use compute_vertical_compensation_area to calculate the bottom boundary area once
      if (const GeometryModel::Box<dim> *gm = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()))
        {
          bottom_boundary_area = dlon;
          this->get_pcout() << "   Bottom boundary area " << bottom_boundary_area << " for dx " << dlon;
          if (dim == 3)
            {
              bottom_boundary_area *= (max_lat - min_lat);
              this->get_pcout() << ", dy " << max_lat - min_lat;
            }
          this->get_pcout() << std::endl;
        }
      else
        {
          bottom_boundary_area = inner_radius * inner_radius * dlon * (std::cos(min_lat) - std::cos(max_lat));
          // TODO adapt for 2D as well
          this->get_pcout() << "   Bottom boundary area " << bottom_boundary_area << " for R,dlon,mincolat,maxcolat " << inner_radius << ", " << dlon / numbers::PI * 180. << ", " << min_lat / numbers::PI * 180. << ", " << max_lat / numbers::PI * 180. << std::endl;
        }

      // Get the boundary velocity objects on the vertical boundaries
      // TODO only keep the objects on lateral boundaries.
      const std::map<types::boundary_id, std::vector<std::unique_ptr<BoundaryVelocity::Interface<dim>>>> &lateral_boundary_velocity_objects =
        this->get_boundary_velocity_manager().get_active_boundary_velocity_conditions();

      std::set<types::boundary_id> tmp_boundary_ids;
      for (const auto &p : lateral_boundary_velocity_objects)
        {
          if (vertical_boundary_indicators.find(p.first) != vertical_boundary_indicators.end())
            tmp_boundary_ids.insert(p.first);
          //lateral_boundary_velocity_objects.erase(p.first);
        }

      AssertThrow(tmp_boundary_ids.size() != 0,
                  ExcMessage("The Compensating Bottom Flow velocity boundary conditions plugin requires prescribed velocity boundary conditions on at least one lateral boundary."));
    }

    template <int dim>
    void
    CompensatingBottomFlow<dim>::update ()
    {
      // Compute the net flow through the lateral boundaries
      net_outflow = compute_net_outflow();
      this->get_pcout() << "    Current net outflow is " << net_outflow << std::endl;
    }


    template <int dim>
    Tensor<1,dim>
    CompensatingBottomFlow<dim>::
    boundary_velocity (const types::boundary_id boundary_indicator,
                       const Point<dim> &position) const
    {
      // We only return a non-zero velocity for the bottom boundary
      if (boundary_indicator != bottom_boundary_indicator)
        return Tensor<1, dim>();

      const Tensor<1, dim>
      upward_normal = -this->get_gravity_model().gravity_vector(position) / this->get_gravity_model().gravity_vector(position).norm();
      return (net_outflow / bottom_boundary_area) * upward_normal;
    }


    template <int dim>
    double
    CompensatingBottomFlow<dim>::
    compute_net_outflow () const
    {
      // Upon initialization, the coarse_mesh is available
      // create a quadrature formula based on the temperature element alone.
      const QGauss<dim-1> quadrature_formula (this->introspection().polynomial_degree.velocities + 1);

      FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                        this->get_fe(),
                                        quadrature_formula,
                                        update_normal_vectors |
                                        update_quadrature_points | update_JxW_values);

      std::map<types::boundary_id, double> local_boundary_fluxes;

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();

      // for every surface face on which it makes sense to compute a
      // mass flux and that is owned by this processor,
      // integrate the normal flux given by the formula
      //   j =  v * n
      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) && vertical_boundary_indicators.find(cell->face(f)->boundary_id())!=vertical_boundary_indicators.end())
              {
                fe_face_values.reinit (cell, f);

                const types::boundary_id id
                  = cell->face(f)->boundary_id();

                typename std::map<types::boundary_id, std::vector<std::unique_ptr<BoundaryVelocity::Interface<dim>>>>::const_iterator boundary_plugins =
                  lateral_boundary_velocity_objects.find(id);

                double local_normal_flux = 0;
                for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                  {
                    for (const auto &plugin : boundary_plugins->second)
                      local_normal_flux += plugin->boundary_velocity(id, fe_face_values.quadrature_point(q)) * fe_face_values.normal_vector(q)
                                           * fe_face_values.JxW(q);
                  }

                local_boundary_fluxes[id] += local_normal_flux;
              }

      // Compute the net flow through the lateral boundaries
      std::vector<double> global_values;
      // now communicate to get the global values
      {
        // first collect local values in the same order in which they are listed
        // in the set of boundary indicators
        std::vector<double> local_values;
        for (std::set<types::boundary_id>::const_iterator
             p = vertical_boundary_indicators.begin();
             p != vertical_boundary_indicators.end(); ++p)
          local_values.push_back (local_boundary_fluxes[*p]);

        global_values.resize(local_values.size());

        // then collect contributions from all processors
        Utilities::MPI::sum (local_values, this->get_mpi_communicator(), global_values);
      }

      const double net_flow = std::accumulate(global_values.begin(), global_values.end(), 0.);

      return net_flow;
    }


    template <int dim>
    void
    CompensatingBottomFlow<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Boundary velocity model");
      {
        prm.enter_subsection ("Compensating bottom flow model");
        {
          prm.declare_entry ("Lateral compensation boundary indicators", "",
                             Patterns::List(Patterns::Selection("east|west|south|north|left|right|front|back")),
                             "A comma separated list of lateral boundary indicators "
                             "specifying which vertical boundaries are used for computing "
                             "a net in/outflow that will be compensated through bottom out/inflow. ");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    CompensatingBottomFlow<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary velocity model");
      {
        prm.enter_subsection("Compensating bottom flow model");
        {
          const std::vector<std::string> x_vertical_boundary_indicators
            = Utilities::split_string_list(prm.get ("Lateral compensation boundary indicators"));
          for (std::vector<std::string>::const_iterator it = x_vertical_boundary_indicators.begin();
               it != x_vertical_boundary_indicators.end(); ++it)
            {
              types::boundary_id boundary_id = numbers::invalid_boundary_id;
              try
                {
                  boundary_id
                    = this->get_geometry_model().translate_symbolic_boundary_name_to_id (*it);

                  vertical_boundary_indicators.insert(boundary_id);
                }
              catch (const std::string &error)
                {
                  AssertThrow (false, ExcMessage ("While parsing the entry <Boundary velocity model/Lateral compensation boundary indicators>, "
                                                  "there was an error. Specifically, "
                                                  "the conversion function complained as follows: "
                                                  + error));
                }
            }


          AssertThrow (vertical_boundary_indicators.size() != 0,
                       ExcMessage("The Compensating Bottom Flow boundary velocity plugin requires at least one lateral boundary to be selected by the user for compensating through bottom flow."));
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
  namespace BoundaryVelocity
  {
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(CompensatingBottomFlow,
                                            "compensating bottom flow",
                                            "Implementation in which the bottom boundary "
                                            "velocity compensates the net in- or outflow through "
                                            "the lateral boundaries.")
  }
}
