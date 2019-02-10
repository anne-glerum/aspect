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


#include <aspect/initial_temperature/plume_only.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/boundary_temperature/interface.h>


#include <cmath>

namespace aspect
{
  namespace InitialTemperature
  {
    template <int dim>
    PlumeOnly<dim>::PlumeOnly ()
    {}

    template <int dim>
    void
    PlumeOnly<dim>::initialize ()
    {
      // verify that the we have a plume boundary temperature model
      // which will give us the plume position
      // TODO check
//      AssertThrow (this->get_boundary_temperature_manager().template has_matching_boundary_temperature_model<BoundaryTemperature::PlumeOnly<dim> >(),
//                  ExcMessage ("This initial temperature is only implemented if the boundary "
//                              "temperature plugin is the 'plume' model."));

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
      else
        AssertThrow(false, ExcNotImplemented());

      // Initialize the plume position lookup
      plume_lookup.reset(new BoundaryTemperature::internal::PlumeOnlyLookup<dim>(plume_data_directory+plume_file_name,
                                                                             this->get_pcout(),
                                                                             cartesian,
                                                                             inner_radius));

    }


    template <int dim>
    double
    PlumeOnly<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      // At the top boundary we do not want to prescribe a plume
      // anomaly.
      // Also, if the boundary conditions call for the inital temperature
      // after t0, we do not want to add the plume anomaly, because that
      // is already done by the plume boundary temperature plugin
      if (this->get_time() > 0.)
      {
        return 0;
      }

      // Adjust the current time for the time after which
      // the plume tail should start
      const Point<dim> plume_position = plume_lookup->plume_position(this->get_time()
                                              - model_time_to_start_plume_tail);

      double initial_temperature(0);

      // Cartesian box
      if (cartesian)
      {
        // Compute plume head or tail temperature
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
          initial_temperature += tail_amplitude * std::exp(-std::pow((position-plume_position).norm()/tail_radius,2));
        }
        else if ((position-plume_position).norm() < current_head_radius)
          initial_temperature += head_amplitude;
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
//        AssertThrow(c1>=0 && c2>=c1, ExcMessage("Point on bottom boundary does not fall in artificial tail cylinder. Distance head to boundary: " + std::to_string(distance_head_to_boundary)
//            + ", position norm: " + std::to_string(position.norm()) + ", inner radius: " + std::to_string(inner_radius) + ", top cylinder: " + std::to_string(top_tail_cylinder_axis.norm())
//            + ", c1 " + std::to_string(c1) + ", c2 " + std::to_string(c2)));
//        const double distance_to_tail_axis = Tensor<1,dim> (position -  c1/c2 * top_tail_cylinder_axis).norm();
        const double distance_to_tail_axis = (cross_product_3d(position,plume_position/inner_radius)).norm();

        // If the two spheres (plume head and inner_radius) intersect,
        // their intersection is a circle with radius current_head_radius.
        // The center of the inner_radius sphere is origin
        const double distance_sphere_centers = tmp_plume_position.norm();
        if (distance_sphere_centers <= head_radius + inner_radius || distance_sphere_centers >= abs(inner_radius - head_radius))
          current_head_radius = std::sqrt(4*inner_radius*inner_radius*distance_sphere_centers*distance_sphere_centers
              -std::pow(distance_sphere_centers*distance_sphere_centers-head_radius*head_radius+inner_radius*inner_radius,2))/(2*distance_sphere_centers);

        //Normal plume tail if most of the plume head has passed
        if ((this->get_time() >= model_time_to_start_plume_tail)
            && (current_head_radius < tail_radius))
        {
          // Prescribe temperature in tail
          initial_temperature += tail_amplitude * std::exp(-std::pow(distance_to_tail_axis/tail_radius,2));
        }
        else if (point_in_plume_head)
        {
          // Prescribe temperature in head
          initial_temperature += head_amplitude;
        }
      }

    return initial_temperature;
    }


    template <int dim>
    void
    PlumeOnly<dim>::declare_parameters (ParameterHandler &prm)
    {
    }


    template <int dim>
    void
    PlumeOnly<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Plume");
      {
        plume_data_directory        = prm.get ("Data directory");
        {
          const std::string      subst_text = "$ASPECT_SOURCE_DIR";
          std::string::size_type position;
          while (position = plume_data_directory.find (subst_text),  position!=std::string::npos)
            plume_data_directory.replace (plume_data_directory.begin()+position,
                plume_data_directory.begin()+position+subst_text.size(),
                ASPECT_SOURCE_DIR);
        }

        plume_file_name                = prm.get ("Plume position file name");
        tail_amplitude = prm.get_double ("Amplitude");
        tail_radius = prm.get_double ("Radius");

        head_amplitude = prm.get_double("Head amplitude");
        head_radius = prm.get_double("Head radius");
        head_velocity = prm.get_double("Head velocity");
        model_time_to_start_plume_tail = prm.get_double ("Model time to start plume tail");

        // convert input ages to seconds
        if (this->convert_output_to_years())
        {
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
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(PlumeOnly,
                                              "plume only",
                                              "Temperature is prescribed as an adiabatic "
                                              "profile with upper and lower thermal boundary layers, "
                                              "whose ages are given as input parameters or as a data file.")
  }
}
