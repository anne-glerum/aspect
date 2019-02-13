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


#include <aspect/global.h>
#include <aspect/boundary_velocity/plume_only.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/ellipsoidal_chunk.h>

#include <deal.II/base/parameter_handler.h>


namespace aspect
{
  namespace BoundaryVelocity
  {

    template <int dim>
    PlumeOnly<dim>::PlumeOnly ()
      :
      boundary_id(numbers::invalid_boundary_id),
      volume(0),
      area(0),
      distance_head_to_boundary(0),
      current_head_radius(0),
      inner_radius(0),
      outer_radius(0),
      cartesian(false),
      boundary_normal_plume_inflow(false)
    {}


    template <int dim>
    void
    PlumeOnly<dim>::initialize ()
    {
      // TODO For now, this plume inflow only works in 3D
      AssertThrow (dim == 3, ExcMessage("This boundary velocity plume plugin has not been checked for 2d."));

      // Check for which boundary indicator this plugin is set up
      const std::map<types::boundary_id,std::vector<std::shared_ptr<BoundaryVelocity::Interface<dim> > > >
      bvs = this->get_boundary_velocity_manager().get_active_boundary_velocity_conditions();
      for (typename std::map<types::boundary_id,std::vector<std::shared_ptr<BoundaryVelocity::Interface<dim> > > >::const_iterator
           p = bvs.begin();
           p != bvs.end(); ++p)
        {
          for (typename std::vector<std::shared_ptr<BoundaryVelocity::Interface<dim> > >::const_iterator
               boundary_plugin = p->second.begin();
               boundary_plugin != p->second.end(); ++boundary_plugin)
            if ((*boundary_plugin).get() == this)
              boundary_id = p->first;
        }

      AssertThrow(boundary_id != numbers::invalid_boundary_id,
                  ExcMessage("Did not find the boundary indicator for the prescribed data plugin."));

//     AssertThrow(this->get_geometry_model().translate_id_to_symbol_name(boundary_id) == "bottom",
//                 ExcMessage("The plume inflow velocity boundary condition can only be set for the bottom boundary."));


      // Set some parameters concerning the domain
      // to compute the area of the bottom boundary
      if (GeometryModel::Box<dim> *gm = dynamic_cast<GeometryModel::Box<dim> *>
                                        (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          cartesian = true;
          const Point<dim> extents = gm->get_extents();
          const Point<dim> origin = gm->get_origin();
          for (unsigned int d = 0; d<dim-1; d++)
            area = extents[0] * extents[dim-2];
          inner_radius = origin[dim-1];
        }
      else if (GeometryModel::Chunk<dim> *gm = dynamic_cast<GeometryModel::Chunk<dim> *>
                                               (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          const double lon_min = gm->west_longitude();
          const double lon_max = gm->east_longitude();
          const double colat_min = 0.5*numbers::PI - gm->north_latitude();
          const double colat_max = 0.5*numbers::PI - gm->south_latitude();
          inner_radius = gm->inner_radius();
          outer_radius = gm->outer_radius();

          // compute the bottom area
          // int int R2 sin(theta) dtheta dphi =
          // R2 int [-cos(theta)]colat_min->colat_max dphi =
          // R2 int [-cos(colat_max) - -cos(colat_min)] dphi =
          // -R2 int [cos(colat_max) - cos(colat_min)] dphi =
          // -R2 (cos(colat_max) - cos(colat_min)) (lon_max-lon_min)
          area = inner_radius * inner_radius * (-std::cos(colat_max) + std::cos(colat_min)) * (lon_max-lon_min);
        }
      else if (GeometryModel::EllipsoidalChunk<dim> *gm = dynamic_cast<GeometryModel::EllipsoidalChunk<dim> *>
                                                          (const_cast<GeometryModel::Interface<dim> *>(&this->get_geometry_model())))
        {
          AssertThrow(gm->get_eccentricity()==0, ExcMessage("This plume boundary velocity plugin does not work for an ellipsoidal domain."));
          const std::vector<Point<2> > corners = gm->get_corners();
          // convert to radians and colatitude
          const double lon_min = corners[2][0]*numbers::PI/180.;
          const double lon_max = corners[0][0]*numbers::PI/180.;
          const double colat_min = (90 - corners[0][1])*numbers::PI/180.;
          const double colat_max = (90 - corners[2][1])*numbers::PI/180.;
          outer_radius = gm->get_semi_major_axis_a();
          inner_radius = outer_radius - gm->maximal_depth();
          area = inner_radius * inner_radius * (std::cos(colat_min) - std::cos(colat_max)) * (lon_max-lon_min);
        }
      // TODO implement for spherical shell
      else
        AssertThrow(false, ExcNotImplemented());


      // Initialize the plume position lookup
      plume_lookup.reset(new BoundaryTemperature::internal::PlumeOnlyLookup<dim>(plume_data_directory+plume_file_name,
                                                                                 this->get_pcout(),
                                                                                 cartesian,
                                                                                 inner_radius));
    }


    template <int dim>
    void
    PlumeOnly<dim>::update ()
    {
      Interface<dim>::update ();

      // Update the plume position
      // (cartesian coordinates, assuming radius at bottom boundary)
      plume_position = plume_lookup->plume_position(this->get_time() - model_time_to_start_plume_tail);

      // reset distance between plume and bottom boundary
      distance_head_to_boundary = 0;
      // reset radius of intersection of plume and bottom boundary
      current_head_radius = 0.;

      // Cartesian box
      if (cartesian)
        {
          // Absolute vertical distance [m] between the center of the plume
          // and the bottom boundary
          distance_head_to_boundary = fabs(head_velocity * (this->get_time() - model_time_to_start_plume_tail));

          // Compute the radius of the plume head in the plane of the bottom boundary
          // if the plume head and bottom boundary intersect
          if (distance_head_to_boundary < head_radius)
            {
              current_head_radius = sqrt(head_radius * head_radius
                                         - distance_head_to_boundary * distance_head_to_boundary);
            }
        }
      // Spherical geometries
      else
        {
          // Radial distance [m] between plume center and bottom boundary
          // in the direction of gravity
          distance_head_to_boundary = head_velocity * (this->get_time() - model_time_to_start_plume_tail);

          // Adapt the plume position radius which was set to the bottom boundary
          const Point<dim> tmp_plume_position = plume_position * (inner_radius + distance_head_to_boundary) / inner_radius;

          // If the two spheres (plume head and inner_radius) intersect,
          // their intersection is a circle with radius current_head_radius.
          // The center of the inner_radius sphere is origin
          const double distance_sphere_centers = tmp_plume_position.norm();
          if (distance_sphere_centers <= head_radius + inner_radius && distance_sphere_centers >= abs(inner_radius - head_radius))
            current_head_radius = std::sqrt(4.*inner_radius*inner_radius*distance_sphere_centers*distance_sphere_centers
                                            -std::pow(distance_sphere_centers*distance_sphere_centers-head_radius*head_radius+inner_radius*inner_radius,2.))/(2.*distance_sphere_centers);
        }

      // Integrate the plume inflow over the bottom boundary
      volume = integrate_plume_inflow();
    }


    template <int dim>
    Tensor<1,dim>
    PlumeOnly<dim>::cartesian_velocity (const Point<dim> position) const
    {
      Tensor<1,dim> velocity;

      // If most of the plume head has passed, use the
      // the tail velocity
      if ((this->get_time() >= model_time_to_start_plume_tail)
          && (current_head_radius < tail_radius))
        {
          // Prescribe velocity in the z direction
          // Gaussian distribution based on the radial distance
          // from the plume center in the plane of the bottom
          // boundary
          velocity[dim-1] += tail_velocity * std::exp(-std::pow((position-plume_position).norm()/tail_radius,2));
        }
      // If the current point lies within the plume head,
      // use the plume velocity
      else if ((position-plume_position).norm() < current_head_radius)
        {
          // Prescribe velocity in the z direction
          velocity[dim-1] += head_velocity;
        }

      return velocity;
    }

    template <int dim>
    Tensor<1,dim>
    PlumeOnly<dim>::
    spherical_velocity (const Point<dim> position) const
    {
      Tensor<1,dim> velocity;

      // Adapt the radial position of the plume based on time passed
      const Point<dim> tmp_plume_position = plume_position * (inner_radius + distance_head_to_boundary) / inner_radius;
      // Compute the distance from the current point to the line axis of the cylinder
      // representing the tail of the plume. For simplicity, the cylinder is extended
      // to the domain's surface
      // These two methods give the same result
      // but the latter also accounts for the case when the plume
      // is on one hemisphere and the point on the other
//      const Point<dim> top_tail_cylinder_axis = tmp_plume_position / tmp_plume_position.norm() * outer_radius;
//      const double c1 = top_tail_cylinder_axis * position;
//      const double c2 = top_tail_cylinder_axis * top_tail_cylinder_axis;
//      AssertThrow(c1>=0 && c2>=c1, ExcMessage("Point on bottom boundary does not fall in artificial tail cylinder. Distance head to boundary: " + std::to_string(distance_head_to_boundary)
//            + ", position norm: " + std::to_string(position.norm()) + ", inner radius: " + std::to_string(inner_radius) + ", top cylinder: " + std::to_string(top_tail_cylinder_axis.norm())
//            + ", c1 " + std::to_string(c1) + ", c2 " + std::to_string(c2)));
//      const double distance_to_tail_axis = Tensor<1,dim> (position -  c1/c2 * top_tail_cylinder_axis).norm();
      const double distance_to_tail_axis = (cross_product_3d(position,plume_position/inner_radius)).norm();

      // Compute the distance between the current point and the center of the plume
      // to see if the point falls within the plume head
      const bool point_in_plume_head = ((position-tmp_plume_position).norm() <= head_radius) ? true : false;

      // Normal plume tail if most of the plume head has passed
      if ((this->get_time() >= model_time_to_start_plume_tail)
          && (current_head_radius < tail_radius))
        {
          // Prescribe velocity in direction parallel to plume
          // (so not in the boundary normal direction except directly
          // above the plume)
          if (boundary_normal_plume_inflow)
            velocity  += tail_velocity * std::exp(-std::pow(distance_to_tail_axis/tail_radius,2)) * position / position.norm();
          else
            velocity  += tail_velocity * std::exp(-std::pow(distance_to_tail_axis/tail_radius,2)) * plume_position / plume_position.norm();
        }
      else if (point_in_plume_head)
        {
          // Prescribe velocity in direction parallel to plume
          // (so not in the boundary normal direction except directly
          // above the plume)
          if (boundary_normal_plume_inflow)
            velocity += head_velocity * position / position.norm();
          else
            velocity += head_velocity * plume_position / plume_position.norm();
        }

      return velocity;
    }

    template <int dim>
    double
    PlumeOnly<dim>::
    integrate_plume_inflow ()
    {
      const QGauss<dim-1> quadrature_formula (this->introspection().polynomial_degree.velocities + 1);

      FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                        this->get_fe(),
                                        quadrature_formula,
                                        update_normal_vectors |
                                        update_q_points       | update_JxW_values);

      double local_normal_flux = 0;

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();

      // Integrate the normal flux given by the formula
      //   j =  v * -n
      // for every surface face on the bottom boundary
      // that is owned by this processor
      // Negate the unit normal vector because it
      // points outwards, while we want to compute
      // the inflow
      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) && cell->face(f)->boundary_id()==this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom"))
              {
                fe_face_values.reinit (cell, f);

                for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                  {
                    local_normal_flux += (cartesian ? cartesian_velocity(fe_face_values.quadrature_point(q)) : spherical_velocity(fe_face_values.quadrature_point(q)))
                                         * -fe_face_values.normal_vector(q)
                                         * fe_face_values.JxW(q);
                  }
              }

      // Compute the global inflow through the bottom boundary
      // by communicating over all processors
      const double global_inflow = Utilities::MPI::sum (local_normal_flux, this->get_mpi_communicator());

      return global_inflow;
    }

    template <int dim>
    Tensor<1,dim>
    PlumeOnly<dim>::
    boundary_velocity (const types::boundary_id boundary_id,
                       const Point<dim> &position) const
    {
      // Make sure we're not prescribing the plume velocity on the
      // top boundary. Also, when the plume has no prescribed velocity,
      // return zero right away.
      if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) == "top"
          || (head_velocity == 0. && tail_velocity == 0.))
        {
          return Tensor<1,dim>();
        }

      // Get plume velocity
      Tensor<1,dim> velocity = cartesian ? cartesian_velocity(position) : spherical_velocity(position);

      // Compensate everywhere on bottom boundary for plume inflow
      // by averaging the integrated plume inflow over the area
      // of the bottom boundary in the boundary normal direction
      velocity = velocity - (volume/area) * position/position.norm();

      return velocity;
    }


    template <int dim>
    void
    PlumeOnly<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Plume");
      {
        prm.declare_entry ("Boundary normal inflow", "false",
                           Patterns::Bool (),
                           "Whether or not to apply the plume inflow velocity normal "
                           "to the bottom boundary or parallel to the direction of the "
                           "center of the plume head.");
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
        tail_velocity                  = prm.get_double ("Inflow velocity");
        tail_radius                    = prm.get_double ("Radius");

        head_radius                    = prm.get_double("Head radius");
        head_velocity                  = prm.get_double("Head velocity");
        model_time_to_start_plume_tail = prm.get_double ("Model time to start plume tail");

        boundary_normal_plume_inflow   = prm.get_bool("Boundary normal inflow");

        if (this->convert_output_to_years() == true)
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
  namespace BoundaryVelocity
  {
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(PlumeOnly,
                                            "plume only",
                                            "This is a velocity plugin that provides plume inflow velocities "
                                            "for the bottom boundary together with a user-defined function "
                                            "to specify any other in- or outflow through the bottom. "
                                            "")
  }
}
