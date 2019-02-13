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

#include <aspect/global.h>
#include <aspect/boundary_temperature/plume_only.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/ellipsoidal_chunk.h>

#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iostream>
#include <utility>
#include <limits>


namespace aspect
{
  namespace BoundaryTemperature
  {

    namespace internal
    {
      template <int dim>
      PlumeOnlyLookup<dim>::PlumeOnlyLookup(const std::string &filename,
                                            const ConditionalOStream &pcout,
                                            const bool cartesian,
                                            const double inner_radius)
      {
        pcout << std::endl << "   Loading Plume position file for boundary temperature: "
              << filename << "." << std::endl << std::endl;

        // Check whether file exists, we do not want to throw
        // an exception in case it does not, because it could be by purpose
        // (i.e. the end of the boundary condition is reached)
        AssertThrow (fexists(filename),
                     ExcMessage (std::string("Plume position file <")
                                 +
                                 filename
                                 +
                                 "> not found!"));

        const double Myr_in_seconds = 1e6 * year_in_seconds;
        const double km_in_m = 1e3;
        const double degrees_to_radians = numbers::PI/180.;

        std::string temp;
        std::ifstream in(filename.c_str(), std::ios::in);
        AssertThrow (in,
                     ExcMessage (std::string("Couldn't open file <") + filename));

        double time,start_time(0),x,y;

        while (in >> time >> x >> y)
          {
            Point<dim> position;
            switch (dim)
              {
                case 2:
                  position(0) = x;
                  break;
                case 3:
                  position(0) = x;
                  position(1) = y;
                  break;
                default:
                  AssertThrow(false,ExcNotImplemented());
                  break;
              }

            // for box only convert coordinates in km to m
            if (cartesian)
              plume_positions.push_back(position*km_in_m);
            // for other geometries,
            // convert spherical coordinates to cartesian
            // 1. from degrees to radians
            // 2. rescale longitude interval from [-pi,pi] to [0,2pi]
            // 3. go from latitude to colatitude
            // 3. set radius to bottom boundary radius
            // 4. convert from spherical to cartesian coordinates
            // TODO needed?
            else
              {
                std_cxx11::array<double,dim> spherical_position;
                spherical_position[0] = inner_radius;
                for (unsigned int d = 1; d<dim; d++)
                  spherical_position[d] = position[d-1] * degrees_to_radians;
                if (spherical_position[1] < 0.)
                  spherical_position[1] += 2.*numbers::PI;
                spherical_position[dim-1] = 0.5 * numbers::PI - spherical_position[dim-1];
                plume_positions.push_back(Utilities::Coordinates::spherical_to_cartesian_coordinates<dim>(spherical_position));
              }

            // Convert the time to SI units
            if (times.size() == 0)
              start_time = time * Myr_in_seconds;
            times.push_back(start_time - time * Myr_in_seconds);
          }
      }

      template <int dim>
      bool
      PlumeOnlyLookup<dim>::fexists(const std::string &filename) const
      {
        std::ifstream ifile(filename.c_str());
        return !(!ifile); // only in c++11 you can convert to bool directly
      }

      template <int dim>
      Point<dim>
      PlumeOnlyLookup<dim>::plume_position(const double time) const
      {
        // If the current time point is before the earliest
        // time in the plume position file, assume the plume
        // has been constant at its earliest value.
        // If the current time point lies after the last entry
        // in the file, assume the plume has stayed constant
        // since that last entry.
        // Otherwise linearly interpolate between the time steps
        // in the file.
        if (time <= times.front())
          return plume_positions.front();
        else if (time >= times.back())
          return plume_positions.back();
        else
          {
            for (unsigned int i = 0; i < times.size() - 1; i++)
              {
                if ((time >= times[i])
                    && (time < times[i+1]))
                  {
                    const double timestep = times[i+1]-times[i];
                    const double time_weight = (time - times[i]) / timestep;

                    return Point<dim> ((1.0 - time_weight) * plume_positions[i]
                                       + time_weight * plume_positions[i+1]);
                  }
              }
            AssertThrow(false,
                        ExcMessage("Did not find time interval for plume location."))
          }
        return Point<dim>();
      }
    }

// -------------------------------------------------

    template <int dim>
    void
    PlumeOnly<dim>::
    initialize ()
    {
      // Set whether the domain is a box or part of a sphere
      // and what the height/radius of the bottom and top boundaries is
      // TODO for now this plugin only works for a box or a chunk
      if (GeometryModel::Box<dim> *gm = dynamic_cast<GeometryModel::Box<dim> *>
                                        (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          cartesian = true;
          inner_radius = gm->get_origin()[dim-1];
          outer_radius = gm->get_extents()[dim-1]+inner_radius;
        }
      else if (GeometryModel::Chunk<dim> *gm = dynamic_cast<GeometryModel::Chunk<dim> *>
                                               (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          inner_radius = gm->inner_radius();
          outer_radius = gm->outer_radius();
        }
      else if (GeometryModel::EllipsoidalChunk<dim> *gm = dynamic_cast<GeometryModel::EllipsoidalChunk<dim> *>
                                                          (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          AssertThrow(gm->get_eccentricity()==0, ExcMessage("This plume boundary velocity plugin does not work for an ellipsoidal domain."));
          outer_radius = gm->get_semi_major_axis_a();
          inner_radius = outer_radius - gm->maximal_depth();
        }
      else
        AssertThrow(false, ExcNotImplemented());

      // Reset the pointer to the plume position lookup
      // The inner radius is used to set the radius of the
      // plume head position (which is not provided in the file)
      // to a reasonable value.
      lookup.reset(new internal::PlumeOnlyLookup<dim>(data_directory+plume_file_name,
                                                      this->get_pcout(),
                                                      cartesian,
                                                      inner_radius));
    }

    template <int dim>
    void
    PlumeOnly<dim>::
    update ()
    {
      // Adjust the current time for the time after which
      // the plume tail should start
      plume_position = lookup->plume_position(this->get_time()
                                              - model_time_to_start_plume_tail);
    }

    template <int dim>
    Point<dim>
    PlumeOnly<dim>::
    get_plume_position () const
    {
      // A getter for other plugins
      return plume_position;
    }


    template <int dim>
    double
    PlumeOnly<dim>::
    boundary_temperature (const types::boundary_id boundary_indicator,
                          const Point<dim>         &position) const
    {
      // Make sure this boundary condition is only used on the bottom boundary
      // and not at t0 if in conjunction with the initial temperature boundary temperature plugin
      // TODO test for initial temperature boundary temperature plugin
      if (this->get_geometry_model().translate_id_to_symbol_name(boundary_indicator) == "top"
          || this->get_time() == 0)
        {
          return 0;
        }

      double boundary_temperature(0);

      // Cartesian box
      if (cartesian)
        {
          // Compute plume head or tail velocity
          double distance_head_to_boundary,current_head_radius(0);
          // radial distance in m
          distance_head_to_boundary = fabs(head_velocity * (this->get_time() - model_time_to_start_plume_tail));

          // If the plume is not yet there, perturbation will not be set
          // Compute the radius of the plume head in the plane of the bottom boundary
          if (distance_head_to_boundary < head_radius)
            {
              current_head_radius = sqrt(head_radius * head_radius
                                         - distance_head_to_boundary * distance_head_to_boundary);
            }

          //Normal plume tail if most of the plume head has passed
          if ((this->get_time() >= model_time_to_start_plume_tail)
              && (current_head_radius < tail_radius))
            {
              // T=T_0*exp-(r/r_0)**2
              boundary_temperature += tail_amplitude * std::exp(-std::pow((position-plume_position).norm()/tail_radius,2));
            }
          else if ((position-plume_position).norm() < current_head_radius)
            boundary_temperature += head_amplitude;
        }
      // Spherical geometries
      else
        {
          double distance_head_to_boundary(0),current_head_radius(0);
          // Does the current point lie in the plume head?
          bool point_in_plume_head = false;
          if ((position-plume_position).norm() <= head_radius)
            point_in_plume_head = true;

          // radial distance in m
          distance_head_to_boundary = head_velocity * (this->get_time() - model_time_to_start_plume_tail);
          // Adapt the plume position radius which was set to the bottom boundary
          const Point<dim> tmp_plume_position = plume_position * (inner_radius + distance_head_to_boundary) / inner_radius;

          // Compute the distance to the axis of the tail
//        Point<dim> top_tail_cylinder_axis = plume_position / plume_position.norm() * outer_radius;
//        const double c1 = top_tail_cylinder_axis * position;
//        const double c2 = top_tail_cylinder_axis * top_tail_cylinder_axis;
//        AssertThrow(c1>=0 && c2>=c1, ExcMessage("Point on bottom boundary does not fall in artificial tail cylinder."));
//        const double distance_to_tail_axis = Tensor<1,dim> (position -  c1/c2 * top_tail_cylinder_axis).norm();
          const double distance_to_tail_axis = (cross_product_3d(position,plume_position/inner_radius)).norm();

          // If the two spheres (plume head and inner_radius) intersect,
          // their intersection is a circle with radius current_head_radius.
          // The center of the inner_radius sphere is origin
          const double distance_sphere_centers = tmp_plume_position.norm();
          if (distance_sphere_centers <= head_radius + inner_radius && distance_sphere_centers >= abs(inner_radius - head_radius))
            current_head_radius = std::sqrt(4*inner_radius*inner_radius*distance_sphere_centers*distance_sphere_centers
                                            -std::pow(distance_sphere_centers*distance_sphere_centers-head_radius*head_radius+inner_radius*inner_radius,2))/(2*distance_sphere_centers);

          //Normal plume tail if most of the plume head has passed
          if ((this->get_time() >= model_time_to_start_plume_tail)
              && (current_head_radius < tail_radius)  && position.norm() < plume_position.norm())
            {
              // Prescribe temperature in tail
              boundary_temperature += tail_amplitude * std::exp(-std::pow(distance_to_tail_axis/tail_radius,2));
            }
          else if (point_in_plume_head)
            {
              // Prescribe temperature in head
              boundary_temperature += head_amplitude;
            }
        }

      return boundary_temperature;
    }


    template <int dim>
    double
    PlumeOnly<dim>::
    minimal_temperature (const std::set<types::boundary_id> &) const
    {
      // Something very hot as we can't set it to the minimum (0) of this plugin,
      // because it will take the minimum with the other plugins, which will be 0
      return 5000;
    }


    template <int dim>
    double
    PlumeOnly<dim>::
    maximal_temperature (const std::set<types::boundary_id> &) const
    {
      // Something very small so that it
      // doesn't affect the maximum of the other
      // plugins
      return 0;
    }

    template <int dim>
    void
    PlumeOnly<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Plume");
      {
        prm.declare_entry ("Data directory",
                           "$ASPECT_SOURCE_DIR/data/boundary-temperature/plume/",
                           Patterns::DirectoryName (),
                           "The name of a directory that contains the model data. This path "
                           "may either be absolute (if starting with a '/') or relative to "
                           "the current directory. The path may also include the special "
                           "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                           "in which the ASPECT source files were located when ASPECT was "
                           "compiled. This interpretation allows, for example, to reference "
                           "files located in the 'data/' subdirectory of ASPECT. ");
        prm.declare_entry ("Plume position file name", "Tristan.sur",
                           Patterns::Anything (),
                           "The file name of the plume position data. "
                           "If the model domain is a box, the data is "
                           "interpreted as the horizontal x and y coordinates [km] "
                           "of the plume head's center point. "
                           "In case of a spherical domain, the coordinates are "
                           "interpreted as the longitude and latitude of the plume head "
                           "center in degrees.");
        prm.declare_entry ("Amplitude", "0",
                           Patterns::Double (),
                           "Amplitude of the plume tail temperature anomaly. Units: K.");
        prm.declare_entry ("Inflow velocity", "0",
                           Patterns::Double (),
                           "Magnitude of the velocity inflow. Units: K.");
        prm.declare_entry ("Radius", "0",
                           Patterns::Double (),
                           "Radius of the plume tail temperature anomaly. Units: m.");
        prm.declare_entry ("Head amplitude", "0",
                           Patterns::Double (),
                           "Amplitude of the plume head temperature anomaly. Units: K.");
        prm.declare_entry ("Head radius", "0",
                           Patterns::Double (),
                           "Radius of the plume head temperature anomaly. Units: m.");
        prm.declare_entry ("Head velocity", "0",
                           Patterns::Double (),
                           "Magnitude of the plume head velocity inflow. Units: m/s or m/yr.");
        prm.declare_entry ("Model time to start plume tail", "0",
                           Patterns::Double (),
                           "Time before the start of the plume position data at which "
                           "the head starts to flow into the model. Units: years or "
                           "seconds.");
      }
      prm.leave_subsection ();

    }


    template <int dim>
    void
    PlumeOnly<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Plume");
      {
        // Get the path to the data files. If it contains a reference
        // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
        // as a #define
        data_directory        = prm.get ("Data directory");
        {
          const std::string      subst_text = "$ASPECT_SOURCE_DIR";
          std::string::size_type position;
          while (position = data_directory.find (subst_text),  position!=std::string::npos)
            data_directory.replace (data_directory.begin()+position,
                                    data_directory.begin()+position+subst_text.size(),
                                    ASPECT_SOURCE_DIR);
        }

        plume_file_name    = prm.get ("Plume position file name");
        tail_velocity = prm.get_double ("Inflow velocity");
        tail_amplitude = prm.get_double ("Amplitude");
        tail_radius = prm.get_double ("Radius");

        head_amplitude = prm.get_double("Head amplitude");
        head_radius = prm.get_double("Head radius");
        head_velocity = prm.get_double("Head velocity");
        model_time_to_start_plume_tail = prm.get_double ("Model time to start plume tail");

        // convert input ages to seconds
        if (this->convert_output_to_years())
          {
            tail_velocity /= year_in_seconds;
            head_velocity /= year_in_seconds;
            model_time_to_start_plume_tail *= year_in_seconds;
          }
      }
      prm.leave_subsection ();

    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryTemperature
  {
    namespace internal
    {
      template class PlumeOnlyLookup<2>;
      template class PlumeOnlyLookup<3>;
    }
    ASPECT_REGISTER_BOUNDARY_TEMPERATURE_MODEL(PlumeOnly,
                                               "plume only",
                                               "A model in which a plume temperature perturbation at the bottom boundary "
                                               "is computed based on a file specifying the plume position over time.")
  }
}
