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


#include "lithostatic_pressure.h"
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <array>

#include <aspect/gravity_model/interface.h>
#include <aspect/boundary_composition/interface.h>
#include <aspect/boundary_temperature/interface.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/initial_temperature/interface.h>
#include <aspect/geometry_model/sphere.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
//#include <aspect/geometry_model/chunk_4.h>
#include "layered_chunk_4.h"
#include "two_merged_chunks.h"
#include <aspect/geometry_model/ellipsoidal_chunk.h>
#include <aspect/geometry_model/box.h>
//#include <aspect/geometry_model/two_merged_boxes.h>

namespace aspect
{
namespace BoundaryTraction
{
template <int dim>
LP<dim>::LP ()
    :
    n_points(1000),
    pressure(n_points, -1)
{}


template <int dim>
void
LP<dim>::initialize()
{
    // Ensure the traction boundary conditions are used
    traction_bi = this->get_traction_boundary_indicators();
    AssertThrow(!traction_bi.empty(), ExcMessage("No need to calculate the lithostatic pressure "
                "if you're not prescribing traction boundary conditions."));

    // Ensure the user-specified representative point lies within the domain
//      const Point<dim> max_dim = this->get_geometry_model().get_extents();
//      for (unsigned int d=0; d<dim; d++)
//        AssertThrow(representative_point[d] <= max_dim[d], ExcMessage("Representative point lies outside domain."));

    // the below is adapted from adiabatic_conditions/initial_profile.cc
    // but we use the initial temperature and composition and only calculate
    // the pressure profile

    // the spacing of the depth profile
    delta_z = this->get_geometry_model().maximal_depth() / (n_points-1);

    const unsigned int n_compositional_fields = this->n_compositional_fields();

    // the pressure at the surface
    pressure[0]    = this->get_surface_pressure();



    // For spherical(-like) domains, do some modifications to the representative point
    Point<dim> spherical_representative_point(representative_point);
    const double degrees_to_radians = dealii::numbers::PI/180;

    // check location of representative point and
    // set radius to surface of domain
    // for spherical domains
    if (const GeometryModel::SphericalShell<dim> *gm = dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()))
    {
        spherical_representative_point[1] *= degrees_to_radians;

        AssertThrow(spherical_representative_point[0] >= gm->inner_radius() &&
                    spherical_representative_point[0] <= gm->outer_radius() &&
                    spherical_representative_point[1] >= 0.0 &&
                    spherical_representative_point[1] <= gm->opening_angle(),
                    ExcMessage("Representative point outside shell domain."));

        spherical_representative_point[0] = gm->outer_radius();
        spherical_representative_point[2] *= degrees_to_radians;
    }
    else if (const GeometryModel::Chunk<dim> *gm = dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model()))
    {
        spherical_representative_point[1] *= degrees_to_radians;

        AssertThrow(spherical_representative_point[0] >= gm->inner_radius() &&
                    spherical_representative_point[0] <= gm->outer_radius() &&
                    spherical_representative_point[1] >= gm->west_longitude() &&
                    spherical_representative_point[1] <= gm->east_longitude(),
                    ExcMessage("Representative point outside chunk domain."));

        if (dim ==3)
        {
            spherical_representative_point[2] *= degrees_to_radians;

            AssertThrow(spherical_representative_point[2] >= gm->south_latitude() &&
                        spherical_representative_point[2] <= gm->north_latitude(),
                        ExcMessage("Representative point outside chunk domain."));
        }

        spherical_representative_point[0] = gm->outer_radius();
    }
//    else if (const GeometryModel::Chunk4<dim> *gm = dynamic_cast<const GeometryModel::Chunk4<dim>*> (&this->get_geometry_model()))
//    {
//        spherical_representative_point[1] *= degrees_to_radians;
//
//        AssertThrow(spherical_representative_point[0] >= gm->inner_radius() &&
//                    spherical_representative_point[0] <= gm->outer_radius() &&
//                    spherical_representative_point[1] >= gm->west_longitude() &&
//                    spherical_representative_point[1] <= gm->east_longitude(),
//                    ExcMessage("Representative point outside chunk domain."));
//
//        if (dim ==3)
//        {
//            spherical_representative_point[2] *= degrees_to_radians;
//
//            AssertThrow(spherical_representative_point[2] >= gm->south_latitude() &&
//                        spherical_representative_point[2] <= gm->north_latitude(),
//                        ExcMessage("Representative point outside chunk domain."));
//        }
//
//        spherical_representative_point[0] = gm->outer_radius();
//    }
    else if (const GeometryModel::LayeredChunk4<dim> *gm = dynamic_cast<const GeometryModel::LayeredChunk4<dim>*> (&this->get_geometry_model()))
    {
        spherical_representative_point[1] *= degrees_to_radians;

        AssertThrow(spherical_representative_point[0] >= gm->inner_radius() &&
                    spherical_representative_point[0] <= gm->outer_radius() &&
                    spherical_representative_point[1] >= gm->west_longitude() &&
                    spherical_representative_point[1] <= gm->east_longitude(),
                    ExcMessage("Representative point outside chunk domain."));

        if (dim ==3)
        {
            spherical_representative_point[2] *= degrees_to_radians;

            AssertThrow(spherical_representative_point[2] >= gm->south_latitude() &&
                        spherical_representative_point[2] <= gm->north_latitude(),
                        ExcMessage("Representative point outside chunk domain."));
        }

        spherical_representative_point[0] = gm->outer_radius();
    }
    else if (const GeometryModel::TwoMergedChunks<dim> *gm = dynamic_cast<const GeometryModel::TwoMergedChunks<dim>*> (&this->get_geometry_model()))
    {
        spherical_representative_point[1] *= degrees_to_radians;

        AssertThrow(spherical_representative_point[0] >= gm->inner_radius() &&
                    spherical_representative_point[0] <= gm->outer_radius() &&
                    spherical_representative_point[1] >= gm->west_longitude() &&
                    spherical_representative_point[1] <= gm->east_longitude(),
                    ExcMessage("Representative point outside chunk domain."));

        if (dim ==3)
        {
            spherical_representative_point[2] *= degrees_to_radians;

            AssertThrow(spherical_representative_point[2] >= gm->south_latitude() &&
                        spherical_representative_point[2] <= gm->north_latitude(),
                        ExcMessage("Representative point outside chunk domain."));
        }

        spherical_representative_point[0] = gm->outer_radius();
    }
//      else if (const GeometryModel::EllipsoidalChunk<dim> *gm = dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model()))
//        {
//    	  // NB: the semi major axis a is only equal to the radius at all surface points if the eccentricity is zero.
//    	  // TODO: check with Menno whether the right angles are passed from geometry plugin
//
//
//          AssertThrow(//spherical_representative_point[0] >= gm->inner_radius() &&
//                      spherical_representative_point[0] <= gm->get_semi_major_axis_a() &&
//                      spherical_representative_point[1] >= gm->minimum_longitude() &&
//                      spherical_representative_point[1] <= gm->maximum_longitude(),
//                      ExcMessage("Representative point outside chunk domain."));
//
//          spherical_representative_point[1] *= degrees_to_radians;
//
//          if (dim ==3)
//            {
//
//              AssertThrow(spherical_representative_point[2] >= gm->minimum_latitude() &&
//                          spherical_representative_point[2] <= gm->maximum_latitude(),
//                          ExcMessage("Representative point outside chunk domain."));
//
//              spherical_representative_point[2] *= degrees_to_radians;
//            }
//
//          spherical_representative_point[0] = gm->get_semi_major_axis_a();
//        }
    else if (const GeometryModel::Sphere<dim> *gm = dynamic_cast<const GeometryModel::Sphere<dim>*> (&this->get_geometry_model()))
    {
        AssertThrow(spherical_representative_point[0] <= gm->radius(), ExcMessage("Representative point outside sphere domain."));

        spherical_representative_point[0] = gm->radius();
        spherical_representative_point[1] *= degrees_to_radians;
        spherical_representative_point[2] *= degrees_to_radians;
    }
    // cartesian geometries
    else if (const GeometryModel::Box<dim> *gm = dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model()))
    {
        const Point<dim> extents = gm->get_extents();

        for (unsigned int d = 0; d < dim; ++d)
            AssertThrow(representative_point[d] <= extents[d], ExcMessage("Representative point outside box domain."));
    }
//    else if (const GeometryModel::TwoMergedBoxes<dim> *gm = dynamic_cast<const GeometryModel::TwoMergedBoxes<dim>*> (&this->get_geometry_model()))
//    {
//        const Point<dim> extents = gm->get_extents();
//
//        for (unsigned int d = 0; d < dim; ++d)
//            AssertThrow(representative_point[d] <= extents[d], ExcMessage("Representative point outside box domain."));
//    }
    else
        AssertThrow(false, ExcNotImplemented());

    // trapezoidal integration
    // set up the input for the density function of the material model
    typename MaterialModel::Interface<dim>::MaterialModelInputs in0(1, n_compositional_fields);
    typename MaterialModel::Interface<dim>::MaterialModelOutputs out0(1, n_compositional_fields);

    // where to calculate the density
    // for spherical domains
    if (dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()) != 0 ||
            dynamic_cast<const GeometryModel::Sphere<dim>*> (&this->get_geometry_model()) != 0 ||
            dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model()) != 0 ||
//            dynamic_cast<const GeometryModel::Chunk4<dim>*> (&this->get_geometry_model()) != 0 ||
            dynamic_cast<const GeometryModel::LayeredChunk4<dim>*> (&this->get_geometry_model()) != 0 ||
            dynamic_cast<const GeometryModel::TwoMergedChunks<dim>*> (&this->get_geometry_model()) != 0
            /*dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model()) != 0*/ )
    {
        // decrease radius with depth increment
        in0.position[0] = spherical_to_cart(spherical_representative_point);
    }
    // and for cartesian domains
    else
    {
        // set z coordinate to appropriate depth
        in0.position[0] = representative_point;
        in0.position[0][dim-1] = this->get_geometry_model().maximal_depth();
    }

    // we need the actual temperature at t0
    in0.temperature[0] = this->get_initial_temperature_manager().initial_temperature(in0.position[0]);
    // the previous pressure
    // it is not used in the density calculation at the moment
    in0.pressure[0] = pressure[0];

    // we need the actual composition at t0
    for (unsigned int c=0; c<n_compositional_fields; ++c)
        in0.composition[0][c] = this->get_initial_composition_manager().initial_composition(in0.position[0], c);

    in0.strain_rate.resize(0); // we do not need the viscosity
    this->get_material_model().evaluate(in0, out0);

    // get the magnitude of gravity. we assume
    // that gravity always points along the depth direction. this
    // may not strictly be true always but is likely a good enough
    // approximation here.
    const double density0 = out0.densities[0];
    const double gravity0 = this->get_gravity_model().gravity_vector(in0.position[0]).norm();

    double sum = delta_z * 0.5 * density0 * gravity0;




    // now integrate pressure downward using the explicit Euler method for simplicity
    // note: p'(z) = rho(p,c,T) * |g| * delta_z
    // for now we'll assume that the t0 temperature and composition fields at the boundary
    // are representative throughout the model time and model domain
    double z;

    for (unsigned int i=1; i<n_points; ++i)
    {
        AssertThrow (i < pressure.size(), ExcMessage(std::string("The current index ")
                     + dealii::Utilities::int_to_string(i)
                     + std::string(" is bigger than the size of the pressure vector ")
                     + dealii::Utilities::int_to_string(pressure.size())));

        // current depth
        z = double(i) * delta_z;

        // set up the input for the density function of the material model
        typename MaterialModel::Interface<dim>::MaterialModelInputs in(1, n_compositional_fields);
        typename MaterialModel::Interface<dim>::MaterialModelOutputs out(1, n_compositional_fields);

        // where to calculate the density

        // for spherical domains
        if (dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model()) != 0 ||
                dynamic_cast<const GeometryModel::Sphere<dim>*> (&this->get_geometry_model()) != 0 ||
                dynamic_cast<const GeometryModel::Chunk<dim>*> (&this->get_geometry_model()) != 0 ||
                dynamic_cast<const GeometryModel::TwoMergedChunks<dim>*> (&this->get_geometry_model()) != 0
                /*dynamic_cast<const GeometryModel::EllipsoidalChunk<dim>*> (&this->get_geometry_model()) != 0*/)
        {
            // decrease radius with depth increment
            spherical_representative_point[0] -= delta_z;
            in.position[0] = spherical_to_cart(spherical_representative_point);
        }
        // and for cartesian domains
        else
        {
            // set z coordinate to appropriate depth
            in.position[0] = representative_point;
            in.position[0][dim-1] = this->get_geometry_model().maximal_depth() - z;
        }

        // we need the actual temperature at t0
        in.temperature[0] = this->get_initial_temperature_manager().initial_temperature(in.position[0]);
        // the previous pressure
        // it is not used in the density calculation at the moment
        in.pressure[0] = pressure[i-1];

        // we need the actual composition at t0
        for (unsigned int c=0; c<n_compositional_fields; ++c)
            in.composition[0][c] = this->get_initial_composition_manager().initial_composition(in.position[0], c);

        in.strain_rate.resize(0); // we do not need the viscosity
        this->get_material_model().evaluate(in, out);

        // get the magnitude of gravity. we assume
        // that gravity always points along the depth direction. this
        // may not strictly be true always but is likely a good enough
        // approximation here.
        const double density = out.densities[0];
        const double gravity = this->get_gravity_model().gravity_vector(in.position[0]).norm();

        // Euler integration
//          pressure[i] = pressure[i-1]
//                        + density * gravity * delta_z;

        // Trapezoid integration
        pressure[i] = sum + delta_z * 0.5 * density * gravity;
        sum += delta_z * density * gravity;

    }

    Assert (*std::min_element (pressure.begin(), pressure.end()) >=
            -std::numeric_limits<double>::epsilon() * pressure.size(),
            ExcInternalError());

}

template <int dim>
Tensor<1,dim>
LP<dim>::
traction (const Point<dim> &p,
          const Tensor<1,dim> &normal) const
{
    // We want to set the normal component to the vertical boundary
    // to the lithostatic pressure, the rest of the traction
    // components are left set to zero
    Tensor<1,dim> traction;

    // assign correct value to traction
    // get the lithostatic pressure from a linear interpolation of
    // the calculated profile

    traction = -get_pressure(p) * normal;
    return traction;
}

template <int dim>
double
LP<dim>::
get_pressure (const Point<dim> &p) const
{
    // depth at which we need the pressure
    const double z = this->get_geometry_model().depth(p);

    if (z >= this->get_geometry_model().maximal_depth())
    {
        Assert (z <= this->get_geometry_model().maximal_depth() + delta_z,
                ExcInternalError());
        // return deepest (last) pressure
        return pressure.back();
    }

    const unsigned int i = static_cast<unsigned int>(z/delta_z);
    Assert ((z/delta_z) >= 0, ExcInternalError());
    Assert (i+1 < pressure.size(), ExcInternalError());

    // now do the linear interpolation
    const double d=1.0+i-z/delta_z;
    Assert ((d>=0) && (d<=1), ExcInternalError());

    // Case 2 and 3
//      std::cout << "Absolute error " << abs((d*pressure[i]+(1-d)*pressure[i+1])-(31526.76978*(2890000.0-p[dim-1])+(1035.936/(2.0*2890000.0))*(2890000.0-p[dim-1])*(2890000.0-p[dim-1]))) << std::endl;
//      std::cout << "Relative error " << ((d*pressure[i]+(1-d)*pressure[i+1])-(31526.76978*(2890000.0-p[dim-1])+(1035.936/(2.0*2890000.0))*(2890000.0-p[dim-1])*(2890000.0-p[dim-1])))/(31526.76978*(2890000.0-p[dim-1])+(1035.936/(2.0*2890000.0))*(2890000.0-p[dim-1])*(2890000.0-p[dim-1])) << std::endl;

    // Case 4
//      const double P_anal = (z <= 360000.0) ?
//                        (32562.70578 * z - 0.0014388 * z * z) :
//                        (1.15361056e10 + 31526.76978 * z - 1.134963712e10);

    // Case 6
//      const double P_anal = (z <= 360000.0) ?
//                            (32562.70578 * z - 0.0014388 * z * z) :
//                            (184810163.5 + 31535.98265 * z - 1.279565217e-5 * z * z);
//
//    const double P_anal = (z<=135937.5) ? 3250.0 * 9.81 * z : 4334027344.0 + 9.81 * 3300.0 * (z - 135937.5);
//    std::cout << "Depth " << z << " Absolute error " << abs((d*pressure[i]+(1-d)*pressure[i+1])-P_anal) << " Relative error " << ((d*pressure[i]+(1-d)*pressure[i+1])-P_anal)/P_anal << std::endl;


    return d*pressure[i]+(1-d)*pressure[i+1];
//      return P_anal;
}

template <int dim>
Point<dim>
LP<dim>::
spherical_to_cart(const Point<dim >sphere_coord) const
{
    // Input: radius, longitude, latitude
    Point<dim> cart_coord;

    switch (dim)
    {
    case 2:
    {
        cart_coord[0] = sphere_coord[0] * std::cos(sphere_coord[1]); // X
        cart_coord[1] = sphere_coord[0] * std::sin(sphere_coord[1]); // Y
        break;
    }
    case 3:
    {
        cart_coord[0] = sphere_coord[0] * std::sin(sphere_coord[2]) * std::cos(sphere_coord[1]); // X
        cart_coord[1] = sphere_coord[0] * std::sin(sphere_coord[2]) * std::sin(sphere_coord[1]); // Y
        cart_coord[2] = sphere_coord[0] * std::cos(sphere_coord[2]); // Z
        break;
    }
    default:
        Assert (false, ExcNotImplemented());
        break;
    }

    return cart_coord;
}

template <int dim>
void
LP<dim>::declare_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("Boundary traction model");
    {
        prm.enter_subsection("Lithostatic pressure");
        {
            prm.declare_entry ("Representative point", "",
                               Patterns::List(Patterns::Double()),
                               "The point where the pressure profile will be calculated. "
                               "Cartesian coordinates when geometry is a box, otherwise enter radius, longitude, "
                               "and in 3D latitude. Unit: m/degrees.");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}


template <int dim>
void
LP<dim>::parse_parameters (ParameterHandler &prm)
{
    unsigned int refinement;
    prm.enter_subsection("Mesh refinement");
    {
        refinement = prm.get_integer("Initial adaptive refinement") + prm.get_integer("Initial global refinement");
    }
    AssertThrow(std::pow(2.0,refinement) <= 1000, ExcMessage("Not enough integration points for this resolution."));

    prm.leave_subsection();
    prm.enter_subsection("Boundary traction model");
    {
        prm.enter_subsection("Lithostatic pressure");
        {
            // The representative point where to calculate the depth profile
            const std::vector<double> rep_point =
                dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Representative point")));
            AssertThrow(rep_point.size() == dim, ExcMessage("Representative point dimensions unequal to model dimensions."));
            for (unsigned int d = 0; d<dim; d++)
                representative_point[d] = rep_point[d];
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
namespace BoundaryTraction
{
ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(LP,
        "lithostatic pressure",
        "Implementation of a model in which the boundary "
        "traction is given in terms of a normal traction component "
        "set to the lithostatic pressure "
        "calculated according to the parameters in section "
        "``Boundary traction model|Lithostatic pressure''. "
        "\n\n"
        "The lithostatic pressure is calculated by integrating "
        "the pressure downward based on the initial composition "
        "and temperature. "
        "\n\n"
        "Note that the tangential velocity component(s) should be set "
        "to zero. ")
}
}
