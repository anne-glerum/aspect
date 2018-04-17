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
#include <aspect/boundary_velocity/euler_poles.h>
#include <aspect/utilities.h>

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
    EulerPoles<dim>::EulerPoles ()
    {}


    template <int dim>
    void
    EulerPoles<dim>::initialize ()
    {
      // Check that we're using a 3D spherical model
      AssertThrow (dim == 3,
                   ExcMessage ("To use the Euler pole boundary velocity plugin, the model "
                       "must be 3D."));

      AssertThrow (/*((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model())) != 0)
                   || */((dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model())) != 0)
                   || ((dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model())) != 0),
                   ExcMessage ("This Euler pole plugin can only be used when using "
                       "a (ellipsoidal) chunk geometry."));

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
            max_lat = numbers::PI;
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
          // If the eccentricity of the EllipsoidalChunk is non-zero, the radius can vary along a boundary,
          // but the maximal depth is the same everywhere and we could calculate a representative pressure
          // profile. However, it requires some extra logic with ellipsoidal
          // coordinates, so for now we only allow eccentricity zero.
          // Using the EllipsoidalChunk with eccentricity zero can still be useful,
          // because the domain can be non-coordinate parallel.
          AssertThrow(gm->get_eccentricity() == 0.0, ExcMessage("This boundary velocity plugin cannot be used with a non-zero eccentricity. "));

          outer_radius = gm->get_semi_major_axis_a();
          inner_radius = outer_radius - gm->maximal_depth();

          // TODO assuming chunk outlines are lat/lon parallel
          std::vector<Point<2> > corners = gm->get_corners();
          // colat
          min_lat = (90. - corners[0][1]) * numbers::PI / 180.;
          max_lat = (90. - corners[2][1]) * numbers::PI / 180.;
          const double max_lon = corners[0][0];
          const double min_lon = corners[1][0];
          dlon = (max_lon - min_lon) * numbers::PI / 180.;
        }
      else
        AssertThrow(false, ExcMessage("The Euler poles boundary velocity plugin does not work for this geometry model."));

      transition_radius = outer_radius - transition_depth;
      transition_radius_max = transition_radius + transition_width;
      transition_radius_min = transition_radius - transition_width;

      area_scale_factor = (outer_radius * outer_radius - transition_radius_max * transition_radius_max) / (transition_radius_min * transition_radius_min - inner_radius * inner_radius);
      transition_area_scale_factor = std::fabs((transition_width/3. + 0.5 * transition_radius) / (transition_width/3. - 0.5 * transition_radius));

      // Compute the area of the bottom boundary
      // as the integral over longitude interval dlon and latitude interval dlat of R0*R0*sin(lat)
      if (bottom_boundary_compensation)
        {
      bottom_boundary_area = inner_radius * inner_radius * dlon * (std::cos(min_lat) - std::cos(max_lat));
      this->get_pcout() << "   Bottom boundary area " << bottom_boundary_area << " for R,dlon,mincolat,maxcolat " << inner_radius << ", " << dlon / numbers::PI * 180. << ", " << min_lat / numbers::PI * 180. << ", " << max_lat / numbers::PI * 180. << std::endl;
        }
    }

    template <int dim>
    void
    EulerPoles<dim>::update ()
    {
      // Compute the net outward flow through the vertical boundaries
      if (bottom_boundary_compensation || vertical_residual_compensation)
        {
          this->get_pcout() << "    Current net outflow is " << net_outflow << std::endl;
        net_outflow = compute_net_outflow();
        }
      // Compute the area over which the Euler pole velocity is prescribed.
      if  (vertical_residual_compensation)
        {
          vertical_compensation_area = compute_vertical_compensation_area();
          this->get_pcout() << "    Current vertical compensation area is " << vertical_compensation_area << std::endl;
        }
    }


    template <int dim>
    Tensor<1,dim>
    EulerPoles<dim>::
    boundary_velocity (const types::boundary_id boundary_indicator,
                       const Point<dim> &position) const
    {
      // Check if pole was provided for this boundary indicator
      if (boundary_indicator == bottom_boundary_indicator && bottom_boundary_compensation)
        {
          return (net_outflow / bottom_boundary_area) * (position / position.norm());
        }
      else if (boundary_indicator != bottom_boundary_indicator)
        {
          Tensor<1,dim> uncompensated_velocity = compute_vertical_boundary_velocity (boundary_indicator, position);
          if (bottom_boundary_compensation)
            return uncompensated_velocity;
          else if (vertical_residual_compensation)
            {
              // above transition radius, return the euler pole velocity for every boundary
              if (position.norm() > transition_radius)
                return uncompensated_velocity;
              // for boundaries that are not used to compensate the prescribed net flow,
              // set velocity to zero below the transition zone.
              else if (vertical_boundary_compensation_indicators.find(boundary_indicator)==vertical_boundary_compensation_indicators.end())
                return Tensor<1,dim>();
              else
                {
                  // R, phi  (lon), theta (colat)
                  std_cxx11::array<double,dim> spherical_position = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                  // inward normal
                  Point<dim> normal;
                  // The east and west boundary of a chunk are planes...
                  if (boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("east"))
                    {
                      normal[0] =   std::sin(spherical_position[1]);
                      normal[1] = - std::cos(spherical_position[1]);
                    }
                  else if (boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("west"))
                    {
                      normal[0] = - std::sin(spherical_position[1]);
                      normal[1] =   std::cos(spherical_position[1]);
                    }
                  // while the north and south boundary are part of a cone.
                  else if (boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("north")
                      || boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("south"))
                    {
                      normal = -2. * position;
                      const double z_cone = outer_radius*std::cos(spherical_position[dim-1]);
                      const double radius_cone_squared = outer_radius * outer_radius - z_cone * z_cone;
                      const double c_squared = radius_cone_squared / (z_cone * z_cone);
                      normal[0] /= -c_squared;
                      normal[1] /= -c_squared;
                      normal /= normal.norm();
                    }
                  // Return an inward pointing normal vector with an average compensating velocity.
                  return normal * net_outflow / vertical_compensation_area;
                }
            }

          const Tensor<1,dim> compensated_velocity = compute_antiparallel_compensation(position) * uncompensated_velocity;

          return compensated_velocity;
        }
      else
        AssertThrow(false, ExcMessage("The boundary velocity plugin Euler poles has reached a combination of bottom boundary velocity and indicator not implemented. Current boundary indicator " + Utilities::int_to_string(int(boundary_indicator)) +
                                      + " and bottom compensation is " + Utilities::int_to_string(int(bottom_boundary_compensation))));

      // we shouldn't get here
      return Tensor<1,dim>();
    }

    template <int dim>
    Tensor<1,dim>
    EulerPoles<dim>::
    compute_vertical_boundary_velocity (const types::boundary_id boundary_indicator,
                       const Point<dim> &position) const
                       {
      const typename std::map<types::boundary_id, std::vector<Point<dim> > >::const_iterator it_pole = boundary_velocities.find(boundary_indicator);
      const typename std::map<types::boundary_id, std::vector<double > >::const_iterator it_transition = boundary_transitions.find(boundary_indicator);

      Point<dim> pole = (it_pole->second)[0];

      // if there's more than 1 pole, we need to transition smoothly between them
      if ((it_pole->second).size()>1)
        {
          // reset pole
          pole = Point<dim>();
      // determine the rotation vector based on position along the boundary in lon or colat
      const std_cxx11::array<double,dim> spherical_position = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

      if (boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("east")
          || boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("west"))
        {
          for (unsigned int d=0; d<it_transition->second.size()-1; ++d)
            {
              // pole is hypertangent combination of poles around transition0
              pole += (0.5-0.5*std::tanh((spherical_position[dim-1]-(it_transition->second)[d])/(numbers::PI/180.)))*(it_pole->second)[d]
                    + (0.5+0.5*std::tanh((spherical_position[dim-1]-(it_transition->second)[d])/(numbers::PI/180.)))*(it_pole->second)[d+1];
            }
        }
      else if (boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("north")
          || boundary_indicator == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("south"))
        {
          for (unsigned int d=0; d<it_transition->second.size()-1; ++d)
            {
              // pole is hypertangent combination of poles around transition
              pole += (0.5-0.5*std::tanh((spherical_position[1]-(it_transition->second)[d])/(numbers::PI/180.)))*(it_pole->second)[d]
                    + (0.5+0.5*std::tanh((spherical_position[1]-(it_transition->second)[d])/(numbers::PI/180.)))*(it_pole->second)[d+1];
            }
        }
      else
        AssertThrow(false, ExcMessage("The boundary velocity plugin Euler poles found a boundary indicator that is not supported"));
      }

      // Compute the cartesian velocity as the cross product of the pole and the point on the boundary
      Tensor <1,dim> euler_velocity = cross_product_3d(pole, position);

      // Scale the velocity with depth around transition zone such that velocity is zero at bottom of vertical boundaries.
      if (bottom_boundary_compensation)
        {
          const double radius = position.norm();
          const double scale  = std::min(1., (radius - inner_radius) / (transition_radius - inner_radius));
          euler_velocity *= scale;
        }
      // Scale the velocity with depth so that it decreases to zero over the transition zone.
      else if (vertical_residual_compensation)
        {
          const double radius = position.norm();
          const double scale  = std::max(0.,std::min(1., (radius - transition_radius) / (transition_radius_max - transition_radius)));
          euler_velocity *= scale;
        }

      return euler_velocity;
    }

    template <int dim>
    double
    EulerPoles<dim>::
    compute_antiparallel_compensation (const Point<dim> &position) const
    {
      // Compute depth and transition scale factor to transition from outflow to inflow
      const double depth = this->get_geometry_model().depth(position);
      const double transition_scale_factor = std::min(1.,std::max(-1.,(depth - transition_depth) * (-1./transition_width)));
      const double radial_scale_factor = outer_radius / position.norm();

      return (transition_scale_factor * scale_factor * radial_scale_factor) *
          ((depth>outer_radius-transition_radius_min) ? area_scale_factor : 1.) *
          ((depth>outer_radius-transition_radius && depth<outer_radius-transition_radius_min) ? transition_area_scale_factor : 1.);
    }

    template <int dim>
    double
    EulerPoles<dim>::
    compute_net_outflow () const
    {
      // Upon initialization, the coarse_mesh is available
      // create a quadrature formula based on the temperature element alone.
       const QGauss<dim-1> quadrature_formula (this->introspection().polynomial_degree.velocities + 1);

       FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                         this->get_fe(),
                                         quadrature_formula,
                                         update_normal_vectors |
                                         update_q_points       | update_JxW_values);

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

                 double local_normal_flux = 0;
                 for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                   {
                     local_normal_flux += compute_vertical_boundary_velocity(id, fe_face_values.quadrature_point(q)) * fe_face_values.normal_vector(q)
                         * fe_face_values.JxW(q);
                   }

                 local_boundary_fluxes[id] += local_normal_flux;
               }

       // Compute the global outflow through the Euler poles boundaries
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
     double
     EulerPoles<dim>::
     compute_vertical_compensation_area () const
     {
       // Upon initialization, the coarse_mesh is available
       // create a quadrature formula based on the velocity element alone.
        const QGauss<dim-1> quadrature_formula (this->introspection().polynomial_degree.velocities + 1);

        FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                          this->get_fe(),
                                          quadrature_formula,
                                          update_q_points       | update_JxW_values);

        std::map<types::boundary_id, double> local_boundary_areas;

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
              if (cell->at_boundary(f) && vertical_boundary_compensation_indicators.find(cell->face(f)->boundary_id())!=vertical_boundary_compensation_indicators.end())
                {
                  fe_face_values.reinit (cell, f);

                  const types::boundary_id id
                    = cell->face(f)->boundary_id();

                  double local_area = 0;
                  for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                    {
                      if (this->get_geometry_model().depth(fe_face_values.quadrature_point(q)) > transition_depth)
                          local_area += fe_face_values.JxW(q);
                    }

                  local_boundary_areas[id] += local_area;
                }

        // Compute the global outflow through the Euler poles boundaries
        std::vector<double> global_values;
        // now communicate to get the global values
        {
          // first collect local values in the same order in which they are listed
          // in the set of boundary indicators
          std::vector<double> local_values;
          for (std::set<types::boundary_id>::const_iterator
               p = vertical_boundary_compensation_indicators.begin();
               p != vertical_boundary_compensation_indicators.end(); ++p)
            local_values.push_back (local_boundary_areas[*p]);

          global_values.resize(local_values.size());

          // then collect contributions from all processors
          Utilities::MPI::sum (local_values, this->get_mpi_communicator(), global_values);
        }

       const double compensation_area = std::accumulate(global_values.begin(), global_values.end(), 0.);

       return compensation_area;
     }


    template <int dim>
    void
    EulerPoles<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Boundary velocity model");
      {
        prm.enter_subsection ("EulerPoles model");
        {
          prm.declare_entry ("Scale factor", "1",
                             Patterns::Double (0),
                             "Scalar factor, which is applied to the boundary velocity. "
                             "You might want to use this to scale the velocities.");
          prm.declare_entry ("Velocity transition depth", "120000",
                             Patterns::Double (0),
                             "Determines the depth at which the outflow of lithospheric material changes to inflow "
                             "of (asthenospheric) material.");
          prm.declare_entry ("Velocity transition width", "10000",
                             Patterns::Double (0),
                             "Determines the half-width with which the outflow of lithospheric material changes to inflow "
                             "of (asthenospheric) material. A linear transition from outflow to inflow is used. ");
          prm.declare_entry ("Compensate net flow", "false",
                             Patterns::Bool (),
                             "Whether or not to compute the net in/outflow over the vertical boundaries for the prescribed "
                             "euler poles and compensate uniformly for this residual orthogonally to the boundaries. ");
          prm.declare_entry ("Vertical compensation boundary indicators", "",
                             Patterns::List(Patterns::Selection("east|west|south|north|left|right|front|back")),
                             "A comma separated list of vertical boundary indicators "
                             "specifying which vertical boundaries are used for compensating "
                             "a net in/outflow. ");
          prm.declare_entry ("Boundary indicator to velocity mappings", "",
                             Patterns::Map (Patterns::Anything(),
                                            Patterns::Anything()),
                                            "A comma separated list of mappings between boundary "
                                            "indicators and the euler pole associated with the "
                                            "boundary indicators. The format for this list is "
                                            "``indicator1 : value1x; value1y; value1z, indicator2 : value2x; value2y; value2z, ...'', "
                                            "where each indicator is a valid boundary indicator "
                                            "(either a number or the symbolic name of a boundary as provided "
                                            "by the geometry model) "
                                            "and each value is a component of the angular velocity vector of that boundary."
                                            "The Euler vector is assumed to be in spherical coordinates: lon [degrees], lat [degrees], rotation rate [degrees/My]." );
          prm.declare_entry ("Boundary indicator to transition mappings", "",
                             Patterns::Map (Patterns::Anything(),
                                            Patterns::Anything()),
                                            "A comma separated list of mappings between boundary "
                                            "indicators and the transition location associated with the "
                                            "Euler pole given for this boundary indicator. The format for this list is "
                                            "``indicator1 : value1_1; value1_2, indicator2 : value2_1, ...'', "
                                            "where each indicator is a valid boundary indicator "
                                            "(either a number or the symbolic name of a boundary as provided "
                                            "by the geometry model) "
                                            "and each value is a latitude or longitude of that boundary. Units: [degrees]" );
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    EulerPoles<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary velocity model");
      {
        prm.enter_subsection("EulerPoles model");
        {
          scale_factor          = prm.get_double ("Scale factor");
          transition_depth = prm.get_double ("Velocity transition depth");
          transition_width = prm.get_double ("Velocity transition width");
          vertical_residual_compensation = prm.get_bool ("Compensate net flow");
          const std::vector<std::string> x_vertical_boundary_indicators
          = Utilities::split_string_list(prm.get ("Vertical compensation boundary indicators"));
          for (std::vector<std::string>::const_iterator it = x_vertical_boundary_indicators.begin();
              it != x_vertical_boundary_indicators.end(); ++it)
            {
              types::boundary_id boundary_id = numbers::invalid_boundary_id;
              try
              {
                  boundary_id
                  = this->get_geometry_model().translate_symbolic_boundary_name_to_id (*it);

                  vertical_boundary_compensation_indicators.insert(boundary_id);
              }
              catch (const std::string &error)
              {
                  AssertThrow (false, ExcMessage ("While parsing the entry <Boundary velocity model/Vertical compensation boundary indicators>, "
                      "there was an error. Specifically, "
                      "the conversion function complained as follows: "
                      + error));
              }
            }

          // get the list of mappings
          const std::vector<std::string> x_boundary_velocities
          = Utilities::split_string_list(prm.get ("Boundary indicator to velocity mappings"));


          for (std::vector<std::string>::const_iterator it = x_boundary_velocities.begin();
              it != x_boundary_velocities.end(); ++it)
            {
              // each entry has the format (white space is optional):
              // <id> : <value1x; value1y; value1z (might have spaces)>
              const std::vector<std::string> parts = Utilities::split_string_list (*it, ':');

              AssertThrow (parts.size() == 2,
                           ExcMessage (std::string("Invalid entry trying to describe boundary "
                               "velocities. Each entry needs to have the form "
                               "<boundary_id : value1x; value1y; value1z>, "
                               "but there is an entry of the form <") + *it + ">"));

              types::boundary_id boundary_id = numbers::invalid_boundary_id;
              try
              {
                  boundary_id
                  = this->get_geometry_model().translate_symbolic_boundary_name_to_id (parts[0]);
                  // Go to the next boundary indicator, as we don't need an Euler pole for the bottom boundary
                  if (parts[0] == "inner" || parts[0] == "bottom")
                    {
                      bottom_boundary_indicator = boundary_id;
                      bottom_boundary_compensation = true;
                      continue;
                    }

                  vertical_boundary_indicators.insert(boundary_id);
              }
              catch (const std::string &error)
              {
                  AssertThrow (false, ExcMessage ("While parsing the entry <Boundary velocity model/Boundary to velocity mappings>, "
                      "there was an error. Specifically, "
                      "the conversion function complained as follows: "
                      + error));
              }

              AssertThrow (boundary_velocities.find(boundary_id) == boundary_velocities.end(),
                           ExcMessage ("Boundary indicator <" + Utilities::int_to_string(boundary_id) +
                                       "> appears more than once in the list of indicators "
                                       "for velocities boundary conditions."));


              // Check if there is one or multiple poles prescribed per boundary
              const std::vector<std::string> x_poles = Utilities::split_string_list (parts[1], '>');
              AssertThrow(x_poles.size()>=1, ExcMessage("Each boundary specified should have at least one Euler pole listed."))
              for (unsigned int p=0; p<x_poles.size(); ++p)
                {
                  const std::vector<std::string> x_pole = Utilities::split_string_list (x_poles[p], ';');
                  AssertThrow (x_pole.size() == dim,
                               ExcMessage (std::string("Invalid entry trying to describe boundary "
                                   "poles. Each pole entry needs to have the form "
                                   "<value1x; value1y; value1z>, "
                                   "but there is an entry of the form <") + x_poles[p] + ">"));

                  std_cxx11::array<double,dim> pole;
                  // go from lon,lat,angular vel (in [degrees] or [degrees/My])
                  // to angular vel [rad/s], lon [rad], lat [rad]
                  pole[0] = Utilities::string_to_double (x_pole[dim-1]) * (numbers::PI / 180.) / (1e6 * year_in_seconds);
                  for (unsigned int d=1; d<dim; ++d)
                    pole[d] = Utilities::string_to_double (x_pole[d-1]) * numbers::PI / 180.;

                  // Then go from latitude to colatitude
                  pole[dim-1] = 0.5*numbers::PI - pole[dim-1];

                  // Transform spherical rotation vector to cartesian
                  const Point<dim> cartesian_pole = Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(pole);

                  boundary_velocities[boundary_id].push_back(cartesian_pole);
                }
            }

          AssertThrow (vertical_boundary_indicators.size() == boundary_velocities.size(),
                       ExcMessage("The number of vertical boundary indicators does not correspond to the number of boundaries for which Euler pole(s) was specified."));
          this->get_pcout() << "   Found " << boundary_velocities.size() << " euler pole velocity boundaries." << std::endl;

          // get the list of transition mappings
          const std::vector<std::string> x_boundary_transitions
          = Utilities::split_string_list(prm.get ("Boundary indicator to transition mappings"));

          for (std::vector<std::string>::const_iterator it = x_boundary_transitions.begin();
              it != x_boundary_transitions.end(); ++it)
            {
              // each entry has the format (white space is optional):
              // <id> : <value1_1> value1_2>etc (might have spaces)>
              const std::vector<std::string> parts = Utilities::split_string_list (*it, ':');

              AssertThrow (parts.size() == 2,
                           ExcMessage (std::string("Invalid entry trying to describe boundary "
                               "transitions. Each entry needs to have the form "
                               "<boundary_id : value1x; value1y; value1z>, "
                               "but there is an entry of the form <") + *it + ">"));

              types::boundary_id boundary_id = numbers::invalid_boundary_id;
              try
              {
                  boundary_id
                  = this->get_geometry_model().translate_symbolic_boundary_name_to_id (parts[0]);


                  if (parts[0] == "inner" || parts[0] == "bottom")
                      {
                        //bottom_boundary_indicator = boundary_id;
                        //bottom_boundary_compensation = true;
                        continue;
                      }
              }
              catch (const std::string &error)
              {
                  AssertThrow (false, ExcMessage ("While parsing the entry <Boundary velocity model/Boundary to transition mapping>, "
                      "there was an error. Specifically, "
                      "the conversion function complained as follows: "
                      + error));
              }

              const std::vector<std::string> x_transitions = Utilities::split_string_list (parts[1], '>');
              for (unsigned int p=0; p<x_transitions.size(); ++p)
                {
                  double transition = 0;
                  // go from lon or lat in [degrees]
                  // to lon [rad] or colat [rad]
                  transition = Utilities::string_to_double (x_transitions[p]) * (numbers::PI / 180.);

                  if (boundary_id == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("east")
                      || boundary_id == this->get_geometry_model().translate_symbolic_boundary_name_to_id ("west"))
                    transition = 0.5*numbers::PI - transition;

                  boundary_transitions[boundary_id].push_back(transition);
                }

            }

         AssertThrow(boundary_velocities.size() == boundary_transitions.size(),
                     ExcMessage("The number of boundaries for which an Euler poles was specified does"
                                "not equal the number of boundaries for which a transition was specified."));

         for (typename std::map<types::boundary_id, std::vector<Point<dim> > >::const_iterator it = boundary_velocities.begin(); it!=boundary_velocities.end(); ++it)
           {
             AssertThrow(it->second.size() == boundary_transitions[it->first].size(),
                         ExcMessage("The number of transitions specified for boundary "
                             + Utilities::int_to_string(it->first)
                             + " does not match the number of Euler poles."));
           }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(bottom_boundary_compensation != bool(vertical_boundary_compensation_indicators.size()),
                  ExcMessage("When using bottom boundary compensation, one cannot also compensate through the bottom. "));
      AssertThrow(vertical_residual_compensation == bool(vertical_boundary_compensation_indicators.size()),
                  ExcMessage("When using vertical boundary compensation, one should specify indicators for the boundaries through which to compensate. "));
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryVelocity
  {
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(EulerPoles,
                                            "euler poles",
                                            "Implementation of a model in which the boundary "
                                            "velocity is derived from Euler poles that describe "
                                            "plate rotations on a sphere.")
  }
}
