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


#include "ascii_T_regions.h"
#include <aspect/utilities.h>
#include <aspect/simulator_access.h>
#include "multicomponent_vp.h"
#include "ascii_data_function.h"
#include "adiabat_ascii.h"
#include "table_vp.h"
#include <aspect/initial_composition/ascii_data.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/geometry_model/interface.h>
#include <fstream>
#include <iostream>
#include <array>

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

template <int dim>
PolygonLookup<dim>::PolygonLookup (const std::vector<unsigned int> n_points_per_polygon,
                                   const std::string &polygon_file,
                                   const ConditionalOStream &pcout)
    :
    n_polygons(n_points_per_polygon.size()),
    n_polygon_points(std::accumulate(n_points_per_polygon.begin(),n_points_per_polygon.end(),0)),
    cartesian_polygons(n_polygons),
    spherical_polygons(n_polygons)

{
    pcout << std::endl << "   Opening polygons file "
          << polygon_file << "." << std::endl << std::endl;

    std::string temp;
    std::ifstream in_polygon(polygon_file.c_str(), std::ios::in);
    AssertThrow (in_polygon,
                 ExcMessage (std::string("Couldn't open grid file <") + polygon_file));

    // Read in all the data point positions (separated by comma)
    char sep;

    for (unsigned int n = 0; n < n_polygons; ++n)
    {
        for (unsigned int p = 0; p < n_points_per_polygon[n]; ++p)
        {

            Point<dim> new_point;
            if (dim == 3)
            {
                if (!(in_polygon >> new_point[0] >> sep >> new_point[1]))
                {
                    AssertThrow (false, ExcMessage (std::string("Reading point ")
                                                    +
                                                    dealii::Utilities::int_to_string(p)
                                                    +
                                                    std::string(" failed. File corrupted? :")
                                                    +
                                                    polygon_file));
                }
                new_point[2] = R_earth;
            }
            else
            {
                if (!(in_polygon >> new_point[0] >> sep >> new_point[1]))
                {
                    AssertThrow (false, ExcMessage (std::string("Reading point ")
                                                    +
                                                    dealii::Utilities::int_to_string(p)
                                                    +
                                                    std::string(" failed. File corrupted? :")
                                                    +
                                                    polygon_file));
                }

            }

            spherical_polygons[n].push_back(new_point);
//                new_point = new_point / new_point.norm() * R_earth;
            cartesian_polygons[n].push_back(spherical_to_cart(new_point));
        }
    }
    pcout << std::endl << "   Loaded "
          << dealii::Utilities::int_to_string(n_polygons)
          << " polygons " << std::endl << std::endl;

}

template <int dim>
std::vector<Point<dim> >
PolygonLookup<dim>::get_cartesian_polygon(const unsigned int polygon_nr) const
{
    Assert(polygon_nr < n_polygons, ExcMessage("The requested cartesian polygon is out of range. "));
    return cartesian_polygons[polygon_nr];
}

template <int dim>
std::vector<Point<dim> >
PolygonLookup<dim>::get_spherical_polygon(const unsigned int polygon_nr) const
{
    Assert(polygon_nr < n_polygons, ExcMessage("The requested cartesian polygon is out of range. "));
    return spherical_polygons[polygon_nr];
}

template <int dim>
Point<dim>
PolygonLookup<dim>::cart_to_spherical(const Point<dim> coord) const
{
    //TODO: case dim =2 and dim=3
    Point<dim> spherical_coord;
    const std::array<double,dim> temp_sph_coord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(coord);
    //reorder to long,lat,rad; convert polar angle to latitudinal angle; and convert to degrees
    spherical_coord[0] = temp_sph_coord[1]*180.0/numbers::PI;
    if (spherical_coord[0] > 180.0)
    	spherical_coord[0] -= 360.0;
    spherical_coord[1] = 90.0 - (temp_sph_coord[2]*180.0/numbers::PI);
    spherical_coord[2] = temp_sph_coord[0];
    return spherical_coord;

}

template <int dim>
Point<dim>
PolygonLookup<dim>::spherical_to_cart(const Point<dim> spherical_coord) const
{
    Point<dim> cart_coord, temp_spherical_coord;

    switch (dim)
    {
    case 2:
        temp_spherical_coord[0] = spherical_coord[1];
        temp_spherical_coord[1] = numbers::PI/2 - (spherical_coord[0]/(180.0/numbers::PI));
        cart_coord[0] = temp_spherical_coord[0] * std::cos(temp_spherical_coord[1]); // X
        cart_coord[1] = temp_spherical_coord[0] * std::sin(temp_spherical_coord[1]); // Y
        break;
    case 3:
        //reorder from long,lat,rad to rad, phi,theta; convert latitudinal angle to polar angle; and convert to radians
        temp_spherical_coord[0] = spherical_coord[2];
        temp_spherical_coord[1] = spherical_coord[0]/(180.0/numbers::PI);
        temp_spherical_coord[2] = numbers::PI/2 - (spherical_coord[1]/(180.0/numbers::PI));
        cart_coord[0] = temp_spherical_coord[0] * std::sin(temp_spherical_coord[2]) * std::cos(temp_spherical_coord[1]); // X
        cart_coord[1] = temp_spherical_coord[0] * std::sin(temp_spherical_coord[2]) * std::sin(temp_spherical_coord[1]); // Y
        cart_coord[2] = temp_spherical_coord[0] * std::cos(temp_spherical_coord[2]); // Z
        break;
    }

    return cart_coord;

}


}



template <int dim>
void
AsciiPip<dim>::initialize()
{
    Assert (dim == 3, ExcMessage ("This initial condition should be used in 3D."));

    this->get_pcout() << "Initializing initial temperature conditions " << std::endl;

    if (n_slab_fields > 0)
        grid_lookup.reset(new internal::SlabGridLookup<dim>(n_slab_fields,n_hor_grid_points,n_ver_grid_points,datadirectory+slab_grid_file_name,this->get_pcout()));

    if (n_polygon_points > 0)
        polygon_lookup.reset(new internal::PolygonLookup<dim>(n_points_per_polygon,datadirectory+polygons_file_name,this->get_pcout()));

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
    const Point<dim> cartesian_coord = polygon_lookup->spherical_to_cart(spherical_coord);
    T_a = this->get_adiabatic_conditions().temperature(cartesian_coord);

    double temperature = T_a;

    // Get the background temperature:
    // The adiabatic temperature at the bottom of the deepest plate
    // is uniformly set up to the depth of the deepest plate
    // Below that an adiabat with/without thermal anomalies from tomography
    // is prescribed.

    if (depth <= max_air_thickness)
        temperature = T_top;
//    else if (depth < Continental_Plate.get_depth_top()+Continental_Plate.get_thickness())
//    	temperature = std::max(T_top,continental_plate_temperature(depth));
    else if (depth > max_lithosphere_depth)
        //temperature = ascii_model->initial_temperature(pos);
        temperature = this->get_initial_temperature_manager().template get_matching_initial_temperature_model<InitialTemperature::AdiabatAscii<dim> >().initial_temperature(pos);

    // if deeper than slab_depth, return temperature right away
    if (depth > max_slab_depth)
    	return std::max(T_top,std::min(temperature, T_bottom));

    // Now get the initial composition conditions for this point
    // According to the plate type attributed to each composition,
    // calculate the temperature if needed.
//    if (InitialComposition::AsciiDataFunction<dim> *cic = dynamic_cast<InitialComposition::AsciiDataFunction<dim> *>
//            (const_cast<InitialComposition::Interface<dim> *>(&this->get_initial_composition())))
    if(this->get_initial_composition_manager().template has_matching_initial_composition_model<InitialComposition::AsciiDataFunction<dim> >())
    {
     const InitialComposition::AsciiDataFunction<dim> &cic = this->get_initial_composition_manager().template get_matching_initial_composition_model<InitialComposition::AsciiDataFunction<dim> >();
        for (unsigned int n = 0; n < n_volumes; n++)
        {
            std::vector<unsigned int>::const_iterator it;
            for (it=fields_per_volume[n].begin(); it!=fields_per_volume[n].end(); it++)
            {
                //const double field_value = cic->initial_composition(pos,*it);
                const double field_value = cic.initial_composition(pos,*it);
                switch (composition_types[n])
                {
                case 0:
                    break;
                case 1:
                    break;
                // Oceanic plate
                case 2:
                    if (field_value == 1)
                        temperature = std::max(T_top, oceanic_plate_temperature(n,depth));
                    break;
                // Continental plate
                case 3:
                    if (field_value == 1)
                        temperature = std::max(T_top, continental_plate_temperature(depth));
                    break;
                // Continental plate with oceanic region
                case 4:
                    if (field_value > 0)
                        temperature = std::max(T_top, mixed_plate_temperature(n,continental_oceanic_plate,pos));
                    break;
                // Oceanic plate with continental region
                case 5:
                    if (field_value > 0)
                        temperature = std::max(T_top, mixed_plate_temperature(n,oceanic_continental_plate,pos));
                    break;
                // Continental plate with Black Sea
                case 6:
                    if (field_value > 0)
                        temperature = std::max(T_top, mixed_plate_temperature(n,continental_blacksea_plate,pos));
                    break;
                // Aegea: Continental plate with Thin Continental Plate
                case 7:
                    if (field_value > 0)
                        temperature = std::max(T_top, mixed_plate_temperature(n,Aegea,pos));
                    break;
                // Nubia: Continental plate with Thin Old Oceanic Plate
                case 8:
                    if (field_value > 0)
                        temperature = std::max(T_top, mixed_plate_temperature(n,Nubia,pos));
                    break;
                // Eurasia: Continental plate with Thin Old Oceanic, Thin Continental and Young Oceanic Plates
                case 9:
                    if (field_value > 0)
                        temperature = std::max(T_top, mixed_plate_temperature(n,Eurasia,pos));
                    break;
                // Slab
                case 10:
                    if (field_value > 0.5) // field_value == 1 due to grid interpolation or field_value > 0?
                    {
                        if (this->get_pre_refinement_step()+initial_global_refinement <= refinement_limit)
                        {
                            temperature = std::max(T_top,temperature-100.0);
                        }
                        else
                        {
                            temperature = std::max(T_top, slab_temperature(n, pos));
                        }
                    }
                    break;
                // Weak zone
                case 11:
                {
                    if (field_value > 0)  // changed from == 1 for NAF region in WZ volume
                    	temperature = std::max(T_top, continental_plate_temperature(depth));
                    break;
                }
                case 12:
                {
                    if (field_value > 0)  // changed from == 1 for NAF region in WZ volume
                    	temperature = std::max(T_top, mixed_plate_temperature(n,Eurasia,pos));
                    break;
                }
                case 13:
                {
                    if (field_value == 1)
                    	temperature = std::max(T_top, oceanic_plate_temperature(n,depth));
                    break;
                }
                default:
                    AssertThrow (false, ExcNotImplemented());
                    break;
                }

            }
        }
    }

    // check for unexpected temperatures
    // if they occur, output their position, but keep running
    if (temperature > T_bottom + 1.0)
        std::cout << " initial T point max " << temperature << " " << pos << std::endl << std::endl;
    if (temperature < T_top - 1.0)
        std::cout << " initial T point min " << temperature << " " << pos << std::endl << std::endl;

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
    Assert(field_nr < n_compositional_fields, ExcMessage("The field number is out of range. "));

    // Thermal diffusivity
    // TODO: is this value representative?
    MaterialModel::TableVp<dim> *mvp = dynamic_cast<MaterialModel::TableVp<dim> *>
                                       (const_cast<MaterialModel::Interface<dim> *>(&this->get_material_model()));
    const double kappa = mvp->reference_thermal_diffusivity();

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

    if (temp < T_top)
    {
        std::cout << "oceanic " << temp << " depth " << depth << " plate top " << plate_depth_top << std::endl;
    }

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

    if (temp < T_top)
    {
        std::cout << "continental T " << temp << " depth " << depth << " plate top " << plate_depth_top << std::endl;
    }

    return temp;
}

template <int dim>
double
AsciiPip<dim>::
mixed_plate_temperature (const unsigned int field_nr, const CompoType plate_type, const Point<dim> pos) const
{
	// NB field_nr here is volume nr!!! TODO
    Assert(plate_type == continental_oceanic_plate ||  plate_type == oceanic_continental_plate || plate_type == continental_blacksea_plate ||
           plate_type == Aegea || plate_type == Nubia || plate_type == Eurasia,
           ExcMessage("Wrong plate type for mixed plate temperature. "));
    Assert(field_nr < n_compositional_fields,
           ExcMessage("The field number is out of range. "));
    for (unsigned int r=0; r<region_nr_per_volume[field_nr].size(); r++)
        Assert(region_nr_per_volume[field_nr][r] != 10, ExcMessage("This field is said to be of a mixed plate type, "
                "but there is no polygon specified. "));


    // depth of point
    const Point<dim> coord = pos;
    const double depth = this->get_geometry_model().depth(coord);

    // get info from base plate (overall plate type)
    const double base_plate_depth = (plate_type == oceanic_continental_plate) ?
                                    Oceanic_Plate.get_depth_top() + Oceanic_Plate.get_thickness() :
                                    Continental_Plate.get_depth_top() + Continental_Plate.get_thickness();

    const double base_plate_top_depth = (plate_type == oceanic_continental_plate) ?
                                        Oceanic_Plate.get_depth_top() :
                                        Continental_Plate.get_depth_top();

    const double base_plate_age = (plate_type == oceanic_continental_plate) ?
                                        Oceanic_Plate.get_age() :
                                        0;

    // get info for each region within the base plate
    double region_depth = 0;
    double region_top_depth = 0;
    double region_age = 0;
    double distance = 0, erf_term = 0, mixed_plate_depth = 0, mixed_plate_top_depth = 0, mixed_plate_temp = 0;
    double base_plate_temp  = 0, region_temp = 0;

    for (unsigned int r=0; r<region_nr_per_volume[field_nr].size(); r++)
    {
     const Plate region_plate_type = plate_type_per_region[region_nr_per_volume[field_nr][r]];
     region_depth = region_plate_type.get_depth_top() + region_plate_type.get_thickness();
     region_top_depth = region_plate_type.get_depth_top();
     region_age = region_plate_type.get_age();

//     std::cout << "Region nr " << r << " of volume " << field_nr << ". Region depth " << region_depth << ", Region top " << region_top_depth << ", Region age " << region_age << std::endl;
//     std::cout << "Region nr " << r << " of volume " << field_nr << ". Region plate type oceanic? " << region_plate_type.is_plate_oceanic() << std::endl;

    // negative gives outside distance, positive inside distance
     distance = distance_to_polygon(region_nr_per_volume[field_nr][r],coord);
     erf_term = erf(distance/60000.0); //60 km, as in Mathematica

     mixed_plate_depth     = 0.5 * region_depth * (1.0 + erf_term) - 0.5 * base_plate_depth * (-1.0 + erf_term);
     mixed_plate_top_depth = 0.5 * region_top_depth * (1.0 + erf_term) - 0.5 * base_plate_top_depth * (-1.0 + erf_term);

    if (depth < mixed_plate_top_depth)
        mixed_plate_temp = T_top;
    else if ( depth > mixed_plate_depth)
        mixed_plate_temp = T_a;
    else
    {

        base_plate_temp       = (plate_type == oceanic_continental_plate) ?
                                oceanic_plate_temperature(field_nr,depth,(mixed_plate_depth-mixed_plate_top_depth),mixed_plate_top_depth,base_plate_age) :
                                continental_plate_temperature(depth,(mixed_plate_depth-mixed_plate_top_depth),mixed_plate_top_depth);
        region_temp                 = (region_plate_type.is_plate_oceanic()) ?
                                      oceanic_plate_temperature(field_nr,depth,(mixed_plate_depth-mixed_plate_top_depth),mixed_plate_top_depth,region_age) :
                                      continental_plate_temperature(depth,(mixed_plate_depth-mixed_plate_top_depth),mixed_plate_top_depth);
        if (r == 0)
          mixed_plate_temp      = 0.5 * region_temp * (1.0 + erf_term) - 0.5 * base_plate_temp * (-1.0 + erf_term);
        else
          mixed_plate_temp      = 0.5 * region_temp * (1.0 + erf_term) - 0.5 * mixed_plate_temp * (-1.0 + erf_term);
    }
    }


    //AssertThrow (mixed_plate_temp >= T_top && mixed_plate_temp < 3000.0, ExcMessage("Mixed plate temperature below 0 or above 3000 K."));
    if (mixed_plate_temp < T_top-1.0 || mixed_plate_temp > 3000.0)
    {
        std::cout << "mixed" << mixed_plate_temp << " depth " << depth << " point " << coord << " type " << plate_type << std::endl;
    }

    return mixed_plate_temp;

}

template <int dim>
double
AsciiPip<dim>::
slab_temperature (const unsigned int field_nr, const Point<dim> coord) const
{
    Assert(field_nr < n_compositional_fields, ExcMessage("Field nr not correct."));
    Assert(slab_nr_per_volume[field_nr] != 10, ExcMessage("This field does not represent a slab. "));

    double temp = 0;


    // Pointer to material model to get material properties
    MaterialModel::TableVp<dim> *mvp = dynamic_cast<MaterialModel::TableVp<dim> *>
                                       (const_cast<MaterialModel::Interface<dim> *>(&this->get_material_model()));

    // We use the adaptation of McKenzie 1970 by Pranger 2014, page 8.

    // The Reynolds number: reynolds = rho_0 * c_P * v * d / (2 * k) = v * d / (2 * kappa)
    // TODO: Are the reference values representative?
    const double reynolds = subduction_vel[slab_nr_per_volume[field_nr]] * slab_thickness[slab_nr_per_volume[field_nr]] / (2.0 * mvp->reference_thermal_diffusivity());

    // The adiabatic term.
    // Because we set the temperature to T_a up to max_lithosphere_depth,
    // our adiabatic profile starts at max_lithosphere_depth,
    // so we subtract max_lithosphere depth from the depth of the current point.
    const double real_depth = this->get_geometry_model().depth(coord);
    const double depth = real_depth - max_lithosphere_depth;
    // TODO: if we change the gravity profile, should we change the magnitude here?
    const double gravity = 9.81;
    const double exp_term = depth * mvp->reference_thermal_expansion_coefficient() * gravity / mvp->reference_specific_heat();

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
//      if (local_coord[1] > slab_thickness[slab_nr_per_volume[field_nr]])
//        std::cout << "Depth " << real_depth << "In-slab depth " << local_coord[1] << " Along slab length " << local_coord[0] << std::endl;
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
double
AsciiPip<dim>::
distance_to_polygon(const unsigned int region_nr, const Point<dim> coord) const
{
    // The region nr could be 10, but then it should not be taken into account
    // For now I assume that each field has at most two regions
    Assert(region_nr < 10, ExcMessage("The region number is out of range. "));

    // Same point in spherical coordinates
    Point<dim> spherical_coord = polygon_lookup->cart_to_spherical(coord);

    // Adjust radius to top of model, i.e. where the polygons are defined
    spherical_coord[dim-1] = R_earth;
    const Point<dim> cartesian_coord = polygon_lookup->spherical_to_cart(spherical_coord);

    // Outside polygon is negative
    double sign = -1.0;
    // Inside polygon is positive
    const bool inside = is_inside_2D_polygon(region_nr,coord);
    if (inside)
    {
        sign = 1.0;
    }

    // Below is thanks to Casper!

    std::vector<Point<dim> > spherical_polygon = polygon_lookup->get_spherical_polygon(region_nr);
    std::vector<Point<dim> > cartesian_polygon = polygon_lookup->get_cartesian_polygon(region_nr);

    const unsigned int n_poly_points = spherical_polygon.size();
    Assert(n_poly_points == cartesian_polygon.size(),ExcMessage("Spherical and Cartesian polygons do not have the same nr of points"));

    // Initialize a vector of distances for each point of the polygon with a very large distance
    std::vector<double> distances(n_poly_points, 1e23);

    // Create another polygon but with all points shifted 1 position to the right
    std::vector<Point<dim> > shifted_spherical_polygon(n_poly_points), shifted_cartesian_polygon(n_poly_points);
    shifted_spherical_polygon[0] = spherical_polygon[n_poly_points-1];
    shifted_cartesian_polygon[0] = cartesian_polygon[n_poly_points-1];
    for (unsigned int i = 0; i < n_poly_points-1; ++i)
    {
        shifted_spherical_polygon[i+1] = spherical_polygon[i];
        shifted_cartesian_polygon[i+1] = cartesian_polygon[i];
    }

    for (unsigned int i = 0; i < n_poly_points; ++i)
    {
        Tensor<1,dim> diff = cartesian_polygon[i] - shifted_cartesian_polygon[i];
        if ((diff.norm() / R_earth) > 1e-23)
        {
            Tensor<1,dim-1> diff_polygons, diff_point_poly;
            diff_polygons[0] = shifted_spherical_polygon[i][0] - spherical_polygon[i][0];
            diff_polygons[1] = shifted_spherical_polygon[i][1] - spherical_polygon[i][1];
            diff_point_poly[0] = spherical_coord[0] - spherical_polygon[i][0];
            diff_point_poly[1] = spherical_coord[1] - spherical_polygon[i][1];

            const double norm = diff_polygons.norm();
            const double radius = (diff_point_poly * (diff_polygons / norm)) / norm;

            // point lies closer to not-shifted polygon point
            if (radius <= 0.0)
                distances[i] = (Tensor<1,dim> (cartesian_polygon[i] - cartesian_coord)).norm();
            // point lies closer to shifted polygon point
            else if (radius >= 1.0)
                distances[i] = (Tensor<1,dim> (shifted_cartesian_polygon[i] - cartesian_coord)).norm();
            else
            {
                // TODO: This dimensional thing is not pretty, fix it!!!
                Point<dim> spher_point;
                spher_point[0] = spherical_polygon[i][0] + radius * diff_polygons[0];
                spher_point[1] = spherical_polygon[i][1] + radius * diff_polygons[1];
                if (dim == 3)
                    spher_point[2] = R_earth;
                // make point Cartesian here
                const Point<dim> cart_point = polygon_lookup->spherical_to_cart(spher_point);

                distances[i] = (Tensor<1,dim> (cart_point - cartesian_coord)).norm();

            }

        }
        else
            AssertThrow(false,ExcNotImplemented());
    }

    const double minimal_distance = *std::min_element(distances.begin(),distances.end());
    return sign * minimal_distance;
}

template <int dim>
bool
AsciiPip<dim>::
is_inside_2D_polygon(const unsigned int polygon_nr, const Point<dim> cart_coord) const
{
    Assert(polygon_nr < n_points_per_polygon.size(),
           ExcMessage(std::string("Polygon number ")
                      + dealii::Utilities::int_to_string(polygon_nr)
                      + std::string(" out of range ")
                      + dealii::Utilities::int_to_string(n_points_per_polygon.size())));

    bool Inside;
    Inside = false;

    unsigned int n;
    unsigned int p = n_points_per_polygon[polygon_nr]-1;

    std::vector<Point<dim> > polygon = polygon_lookup->get_spherical_polygon(polygon_nr);
    // TODO: this is also done in the calling function --> double
    const Point<dim> spherical_coord = polygon_lookup->cart_to_spherical(cart_coord);
    // TODO: find a prettier way to do this. If it were a vector is was easier :(
    Point<2> coord;
    coord[0] = spherical_coord[0];
    coord[1] = spherical_coord[1];

    //calculate whether or not the point is in the polygon. It lies inside the polygon when on both sides
    //of the point there are an odd number of polygon sides crossed. This code is taken from
    //www.alienryderflex.com/polygon/
    for (n = 0; n < n_points_per_polygon[polygon_nr]; n++)
    {
        Point<2> point_n, point_p;
        point_n[0] = polygon[n][0];
        point_n[1] = polygon[n][1];
        point_p[0] = polygon[p][0];
        point_p[1] = polygon[p][1];

        if ((point_n[1] <=  coord[1]  && point_p[1] >= coord[1]) ||
                (point_p[1] <=  coord[1]  && point_n[1] >= coord[1]))
        {
            if (point_n[0]
                    + (coord[1] - point_n[1])
                    / (point_p[1] - point_n[1])
                    * (point_p[0] - point_n[0])
                    > coord[0])
            {
                Inside = !Inside;
            }
        }
        p=n;
    }

    return Inside;
}



template <int dim>
void
AsciiPip<dim>::declare_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("Initial temperature model");
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
            prm.declare_entry("Data directory", "$ASPECT_SOURCE_DIR/data/initial-conditions/pip/",
                              Patterns::DirectoryName (),
                              "The path to the model data. ");

            prm.declare_entry ("Slab grid file name", "grid_coord.dat",
                               Patterns::Anything(),
                               "The file name of the slab grid points "
                               "of a 3D varying slab surface.");

            prm.declare_entry ("Polygon region file name", "polygon_coord.dat",
                               Patterns::Anything(),
                               "The file name of the polygon region points "
                               "that denote regions of different plate type.");

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

            prm.declare_entry ("Number of points per region", "",
                               Patterns::List (Patterns::Integer (0)),
                               "This parameter specifies the number of polygon points "
                               "for each region polygon.");

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

            prm.declare_entry ("Field numbers per volume", "",
                               Patterns::List (Patterns::List(Patterns::Integer (0),0,Patterns::List::max_int_value,","),
                            		   0,Patterns::List::max_int_value,";"),
                               "This parameter lists the number of the fields, starting at 0, that are present in a volume (plate)."
                               "Volumes are separated by semicolons, fields by commas.");

            prm.declare_entry ("Region numbers per volume", "",
                               Patterns::List (Patterns::List(Patterns::Integer (0),0,Patterns::List::max_int_value,","),
                            		   0,Patterns::List::max_int_value,";"),
                               "This parameter lists the number of the regions, starting at 0, that are present in a volume (plate)."
                               "Volumes are separated by semicolons, regions by commas.");

            prm.declare_entry ("Region number per field", "",
                               Patterns::List (Patterns::Integer (0)),
                               "This parameter lists the number of the region as in the order "
                               "of the polygon file, starting at 0, if a field represents a slab."
                               "Otherwise set number to 10.");
            prm.declare_entry ("Plate type per region", "",
                               Patterns::List (Patterns::Selection("Oceanic plate|Continental plate|Young oceanic plate|Thin continental plate|EEC plate|Old oceanic plate")),
                               "A list of the plate type of each region. ");
            prm.declare_entry ("List of compositional types", "",
                               Patterns::List (Patterns::Selection("Slab|Oceanic plate|Continental plate|Continental oceanic plate|Oceanic continental plate|Continental black sea plate|Continental weak zone|Mixed continental weak zone|Oceanic weak zone|Air|Mantle|Aegea|Nubia|Eurasia")),
                               "A list of types for each composition needed to determine initial T.");

            prm.declare_entry("Oceanic plate top depth", "10500",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the oceanic plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("Thin old oceanic plate top depth", "10500",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the oceanic plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("Old oceanic plate top depth", "10500",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the oceanic plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("Young oceanic plate top depth", "10500",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the oceanic plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("Continental plate top depth", "8200",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the continental plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("Thin continental plate top depth", "8200",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the continental plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("EEC plate top depth", "8200",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the continental plate with "
                              "respect to the actual top of the domain. ");
            prm.declare_entry("Black sea plate top depth", "11000",
                              Patterns::Double(0),
                              "This parameter specifies the depth of the top of the black sea plate with "
                              "respect to the actual top of the domain. ");

            prm.declare_entry("Refinement limit for approximation", "5",
                              Patterns::Integer(0),
                              "This parameter specifies the refinement level to which composition and "
                              "temperature should be approximated by bounding boxes. ");

            prm.declare_entry("Include EEC", "true",
            		          Patterns::Bool(),
            		          "This parameter specifies whether to include the thick East European craton (EEC), "
            		          "or to set this regions thickness to default Continental Plate thickness.");
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
    this->get_pcout() << "Parsing initial temperature conditions parameters " << std::endl;

    // Plate parameters
    Oceanic_Plate.set_thickness(60e3);
    Oceanic_Plate.set_age(21e6*year_in_seconds);
    Oceanic_Plate.set_plate_type(true);

    Young_Oceanic_Plate.set_thickness(30e3);
    Young_Oceanic_Plate.set_age(10e6*year_in_seconds);
    Young_Oceanic_Plate.set_plate_type(true);

    Thin_Old_Oceanic_Plate.set_thickness(120e3);
    Thin_Old_Oceanic_Plate.set_age(200e6*year_in_seconds);
    Thin_Old_Oceanic_Plate.set_plate_type(true);

    Thick_Old_Oceanic_Plate.set_thickness(120e3);
    Thick_Old_Oceanic_Plate.set_age(200e6*year_in_seconds);
    Thick_Old_Oceanic_Plate.set_plate_type(true);

    Old_Oceanic_Plate.set_thickness(120e3);
    Old_Oceanic_Plate.set_age(200e6*year_in_seconds);
    Old_Oceanic_Plate.set_plate_type(true);

    Continental_Plate.set_thickness(120e3);
    Continental_Plate.set_age(100e6*year_in_seconds);

    Thin_Continental_Plate.set_thickness(60e3);
    Thin_Continental_Plate.set_age(100e6*year_in_seconds);

    Anatolia_Plate.set_thickness(70e3);
    Anatolia_Plate.set_age(100e6*year_in_seconds);

    Black_Sea_Plate.set_thickness(75e3);
    Black_Sea_Plate.set_age(100e6*year_in_seconds);


    EEC_Plate.set_age(100e6*year_in_seconds);
    if (include_EEC)
    	EEC_Plate.set_thickness(200e3);
    else
        EEC_Plate.set_thickness(120e3);

    prm.enter_subsection("Initial temperature model");
    {
        prm.enter_subsection("Pip");
        {
            Oceanic_Plate.set_depth_top(prm.get_double ("Oceanic plate top depth"));
            Thin_Old_Oceanic_Plate.set_depth_top(prm.get_double ("Thin old oceanic plate top depth"));
            Old_Oceanic_Plate.set_depth_top(prm.get_double ("Old oceanic plate top depth"));
            Young_Oceanic_Plate.set_depth_top(prm.get_double ("Young oceanic plate top depth"));
            Continental_Plate.set_depth_top(prm.get_double ("Continental plate top depth"));
            Thin_Continental_Plate.set_depth_top(prm.get_double ("Thin continental plate top depth"));
            EEC_Plate.set_depth_top(prm.get_double ("EEC plate top depth"));
            Black_Sea_Plate.set_depth_top(prm.get_double ("Black sea plate top depth"));
        }
        prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Compositional fields");
    n_compositional_fields = prm.get_integer ("Number of fields");
    prm.leave_subsection ();

    prm.enter_subsection("Initial temperature model");
    {
        prm.enter_subsection("Pip");
        {
            // We use another plugin for the background temperature
            AssertThrow( prm.get("Base model") != "pip",
                         ExcMessage("You may not use ``pip'' as the base model for "
                                    "a this model model.") );

            //ascii_model.reset(InitialTemperature::create_initial_conditions<dim>(prm.get("Base model")));
            //if ( SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(ascii_model.get()))
            //    sim->initialize_simulator (this->get_simulator());

            // Where we get our data files for composition
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
            polygons_file_name = prm.get ("Polygon region file name");

            // Refinement limit
            refinement_limit = prm.get_integer("Refinement limit for approximation");

            // Slab information
            n_slab_fields       = prm.get_integer ("Number of slabs");
            max_slab_depth      = prm.get_double("Slab max depth");
            AssertThrow (n_slab_fields <= n_compositional_fields, ExcMessage("The number of slabs is greater than the number of fields. "));

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

            // Region polygon information
            const std::vector<int> n_point = dealii::Utilities::string_to_int
                                             (dealii::Utilities::split_string_list(prm.get("Number of points per region")));
            n_points_per_polygon = std::vector<unsigned int> (n_point.begin(),
                                   n_point.end());

            n_polygon_points = std::accumulate(n_points_per_polygon.begin(),n_points_per_polygon.end(),0);
 
            n_regions = n_points_per_polygon.size();

            // Volume information
            std::vector<std::string> types = dealii::Utilities::split_string_list(prm.get("List of compositional types"));
            AssertThrow(types.size() <= n_compositional_fields,
                        ExcMessage(std::string("Length of compo type list should be smaller or equal to the number of fields instead of")
                                   + dealii::Utilities::int_to_string(types.size())));

            n_volumes = types.size();

            // New 21/11/16
          const std::vector<std::string> region_nrs_all = dealii::Utilities::split_string_list(prm.get("Region numbers per volume"),';');
          AssertThrow (region_nrs_all.size() == n_volumes, ExcMessage("The number of volumes for which a region is specified "
                                                                 "does not correspond to the number of volumes. "));
          region_nr_per_volume.resize(n_volumes);
          for (unsigned int v=0; v<region_nrs_all.size(); v++)
          {
        	  const std::vector<std::string> region_nrs = dealii::Utilities::split_string_list(region_nrs_all[v],',');
        	  std::vector<unsigned int> region_nrs_per_volume;
        	  for (unsigned int d=0; d<region_nrs.size(); d++)
        		  region_nrs_per_volume.push_back(dealii::Utilities::string_to_int(region_nrs[d]));
        	  region_nr_per_volume[v] = region_nrs_per_volume;
          }

          const std::vector<int> region_nrs_allfields = dealii::Utilities::string_to_int(dealii::Utilities::split_string_list(prm.get("Region number per field")));
          AssertThrow (region_nrs_allfields.size() == n_compositional_fields, ExcMessage("The number of fields for which a region is specified "
                                                                 "does not correspond to the number of fields. "));
          region_nr_per_field = std::vector<unsigned int> (region_nrs_allfields.begin(),
        		  region_nrs_allfields.end());

          const std::vector<std::string> type_regions = dealii::Utilities::split_string_list(prm.get("Plate type per region"),',');
          AssertThrow (type_regions.size() == n_regions, ExcMessage("The number of regions for which a type is specified, does not correspond to the number of regions. "));

          plate_type_per_region.resize(n_regions);
          for (unsigned int r=0; r<n_regions; r++)
          {
           if (type_regions[r] == "Oceanic plate")
             {
              plate_type_per_region[r] = Oceanic_Plate;
             }
             else if (type_regions[r] == "Continental plate")
             {
              plate_type_per_region[r] = Continental_Plate;
             }
             else if (type_regions[r] == "Young oceanic plate")
             {
              plate_type_per_region[r] = Young_Oceanic_Plate;
             }
             else if (type_regions[r] == "Thin continental plate")
             {
              plate_type_per_region[r] = Thin_Continental_Plate;
             }
             else if (type_regions[r] == "EEC plate")
             {
              plate_type_per_region[r] = EEC_Plate;
             }
             else if (type_regions[r] == "Old oceanic plate")
             {
              plate_type_per_region[r] = Thin_Old_Oceanic_Plate;
             }
            else
            {
                AssertThrow (false, ExcNotImplemented());
            }
          } 
          const std::vector<std::string> field_nrs_all = dealii::Utilities::split_string_list(prm.get("Field numbers per volume"),';');
          AssertThrow (field_nrs_all.size() == n_volumes, ExcMessage("The number of volumes for which a field is specified "
                                                                 "does not correspond to the number of volumes. "));
          fields_per_volume.resize(n_volumes);
          for (unsigned int f=0; f<field_nrs_all.size(); f++)
          {
        	  const std::vector<std::string> field_nrs = dealii::Utilities::split_string_list(field_nrs_all[f],',');
        	  std::vector<unsigned int> field_nrs_per_volume;
        	  for (unsigned int d=0; d<field_nrs.size(); d++)
        		  field_nrs_per_volume.push_back(dealii::Utilities::string_to_int(field_nrs[d]));
        	  fields_per_volume[f] = field_nrs_per_volume;
          }

          // end new 21/11/16

            const std::vector<int> slab_nr = dealii::Utilities::string_to_int
                                             (dealii::Utilities::split_string_list(prm.get("Slab number per field")));
            AssertThrow (slab_nr.size() == n_volumes, ExcMessage("The number of volumes for which a slab is specified "
                         "does not correspond to the number of volumes. "));
            slab_nr_per_volume = std::vector<unsigned int> (slab_nr.begin(),
                                 slab_nr.end());


            // The type of composition for each volume
            for (unsigned int i = 0; i < n_volumes; i++)
            {
                if (types[i] == "Mantle")
                {
                    composition_types.push_back(mantle);
                }
                else if (types[i] == "Air")
                {
                    composition_types.push_back(air);
                }
                else if (types[i] == "Oceanic plate")
                {
                    composition_types.push_back(oceanic_plate);
                }
                else if (types[i] == "Continental plate")
                {
                    composition_types.push_back(continental_plate);
                }
                else if (types[i] == "Continental oceanic plate")
                {
                    composition_types.push_back(continental_oceanic_plate);
                }
                else if (types[i] == "Oceanic continental plate")
                {
                    composition_types.push_back(oceanic_continental_plate);
                }
                else if (types[i] == "Continental black sea plate")
                {
                    composition_types.push_back(continental_blacksea_plate);
                }
                else if (types[i] == "Aegea")
                {
                    composition_types.push_back(Aegea);
                }
                else if (types[i] == "Nubia")
                {
                    composition_types.push_back(Nubia);
                }
                else if (types[i] == "Eurasia")
                {
                    composition_types.push_back(Eurasia);
                }
                else if (types[i] == "Slab")
                {
                    composition_types.push_back(slab);
                }
                else if (types[i] == "Continental weak zone")
                {
                    composition_types.push_back(continental_weak_zone);
                }
                else if (types[i] == "Mixed continental weak zone")
                {
                    composition_types.push_back(mixed_continental_weak_zone);
                }
                else if (types[i] == "Oceanic weak zone")
                {
                    composition_types.push_back(oceanic_weak_zone);
                }
                else
                {
                    AssertThrow (false, ExcNotImplemented());
                }
            }


            include_EEC = prm.get_bool("Include EEC");
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




    // Calculate the maximum air and lithosphere thickness
    // based on the plates selected in the input file.
    max_air_thickness = 0.0;
    max_lithosphere_depth = 0.0;

    for (unsigned int i = 0; i < n_volumes; i++)
    {
        switch (composition_types[i])
        {
        case 0:
        case 1:
        case 2:
            max_air_thickness = std::max(max_air_thickness, Oceanic_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Oceanic_Plate.get_depth_top()+Oceanic_Plate.get_thickness());
            this->get_pcout() << i << " is an oceanic plate" << std::endl;
            break;
        case 3:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            this->get_pcout() << i << " is an continental plate" << std::endl;
            break;
        case 4:
        case 5:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Oceanic_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Oceanic_Plate.get_depth_top()+Oceanic_Plate.get_thickness());
            break;
        case 6:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Black_Sea_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Black_Sea_Plate.get_depth_top()+Black_Sea_Plate.get_thickness());
            break;
        case 7:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Thin_Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Thin_Continental_Plate.get_depth_top()+Thin_Continental_Plate.get_thickness());
            break;
        case 8:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Thin_Old_Oceanic_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Thin_Old_Oceanic_Plate.get_depth_top()+Thin_Old_Oceanic_Plate.get_thickness());
            break;
        case 9:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Thin_Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Thin_Continental_Plate.get_depth_top()+Thin_Continental_Plate.get_thickness());
//            max_air_thickness = std::max(max_air_thickness, EEC_Plate.get_depth_top());
//            max_lithosphere_depth = std::max(max_lithosphere_depth, EEC_Plate.get_depth_top()+EEC_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Young_Oceanic_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Young_Oceanic_Plate.get_depth_top()+Young_Oceanic_Plate.get_thickness());
            max_air_thickness = std::max(max_air_thickness, Thin_Old_Oceanic_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Thin_Old_Oceanic_Plate.get_depth_top()+Thin_Old_Oceanic_Plate.get_thickness());
            break;
        case 10:
            break;
        case 11:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            break;
        case 12:
            max_air_thickness = std::max(max_air_thickness, Continental_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Continental_Plate.get_depth_top()+Continental_Plate.get_thickness());
            break;
        case 13:
            max_air_thickness = std::max(max_air_thickness, Oceanic_Plate.get_depth_top());
            max_lithosphere_depth = std::max(max_lithosphere_depth, Oceanic_Plate.get_depth_top()+Oceanic_Plate.get_thickness());
            break;
        default:
            break;
        }
    }

    // Output the maximum air and lithosphere thickness as a check.
    this->get_pcout() << "Max air thickness " << max_air_thickness << std::endl;
    this->get_pcout() << "Max lithosphere depth " << max_lithosphere_depth << std::endl;

    // Check that the depth from where temperature anomalies from tomography
    // are added to the geotherm is bigger than the maximum lithosphere thickness
    prm.enter_subsection("Initial temperature model");
    {
        prm.enter_subsection("AdiabatAscii");
        {
            if (prm.get_bool("Add temperature perturbations"))
            {
                const double T_perturb_depth = prm.get_double("Top smoothing depth") - prm.get_double("Top smoothing width");
                AssertThrow(max_lithosphere_depth < T_perturb_depth, ExcMessage("Temperature perturbations added in lithosphere."));
            }
        }
        prm.leave_subsection ();
    }
    prm.leave_subsection ();

    // Get initial global refinement to compare with refinement limit for approximation
    prm.enter_subsection("Mesh refinement");
    {
        initial_global_refinement = prm.get_integer("Initial global refinement");
    }
    prm.leave_subsection ();

    // parse the ascii model parameters
    //ascii_model->parse_parameters(prm);
    //ascii_model->initialize();

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
