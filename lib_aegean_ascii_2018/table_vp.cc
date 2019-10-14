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


#include "table_vp.h"
#include <aspect/simulator_access.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/base/table.h>
#include <fstream>
#include <iostream>
#include "ascii_T_regions.h"

using namespace dealii;

namespace aspect
{
namespace MaterialModel
{


namespace internal
{

class MaterialLookup
{
public:
    MaterialLookup(const std::string &filename,
                   const bool interpol)
    {

        /* Initializing variables */
        interpolation = interpol;
        delta_press=-1.0;
        min_press=-1.0;
        delta_temp=-1.0;
        min_temp=-1.0;
        numtemp=0;
        numpress=0;

        std::string temp;
        std::ifstream in(filename.c_str(), std::ios::in);
        AssertThrow (in,
                     ExcMessage (std::string("Couldn't open file <") + filename));

        getline(in, temp); // eat first line
        getline(in, temp); // eat next line
        getline(in, temp); // eat next line
        getline(in, temp); // eat next line

        in >> min_temp;
        getline(in, temp);
        in >> delta_temp;
        getline(in, temp);
        in >> numtemp;
        getline(in, temp);
        getline(in, temp);
        in >> min_press;
        min_press *= 1e5;  // conversion from [bar] to [Pa]
        getline(in, temp);
        in >> delta_press;
        delta_press *= 1e5; // conversion from [bar] to [Pa]
        getline(in, temp);
        in >> numpress;
        getline(in, temp);
        getline(in, temp);
        getline(in, temp);

        AssertThrow(min_temp >= 0.0, ExcMessage("Read in of Material header failed (mintemp)."));
        AssertThrow(delta_temp > 0, ExcMessage("Read in of Material header failed (delta_temp)."));
        AssertThrow(numtemp > 0, ExcMessage("Read in of Material header failed (numtemp)."));
        AssertThrow(min_press >= 0, ExcMessage("Read in of Material header failed (min_press)."));
        AssertThrow(delta_press > 0, ExcMessage("Read in of Material header failed (delta_press)."));
        AssertThrow(numpress > 0, ExcMessage("Read in of Material header failed (numpress)."));


        max_temp = min_temp + (numtemp-1) * delta_temp;
        max_press = min_press + (numpress-1) * delta_press;

        density_values.reinit(numtemp,numpress);
        thermal_expansivity_values.reinit(numtemp,numpress);
        specific_heat_values.reinit(numtemp,numpress);
        vp_values.reinit(numtemp,numpress);
        vs_values.reinit(numtemp,numpress);
        enthalpy_values.reinit(numtemp,numpress);

        unsigned int i = 0;
        while (!in.eof())
        {
            double temp1,temp2;
            double rho,alpha,cp,vp,vs,h;
            in >> temp1 >> temp2;
            in >> rho;
            if (in.fail())
            {
                in.clear();
                rho = density_values[(i-1)%numtemp][(i-1)/numtemp];
            }
            in >> alpha;
            if (in.fail())
            {
                in.clear();
                alpha = thermal_expansivity_values[(i-1)%numtemp][(i-1)/numtemp];
            }
            in >> cp;
            if (in.fail())
            {
                in.clear();
                cp = specific_heat_values[(i-1)%numtemp][(i-1)/numtemp];
            }
            in >> vp;
            if (in.fail())
            {
                in.clear();
                vp = vp_values[(i-1)%numtemp][(i-1)/numtemp];
            }
            in >> vs;
            if (in.fail())
            {
                in.clear();
                vs = vs_values[(i-1)%numtemp][(i-1)/numtemp];
            }
            in >> h;
            if (in.fail())
            {
                in.clear();
                h = enthalpy_values[(i-1)%numtemp][(i-1)/numtemp];
            }

            getline(in, temp);
            if (in.eof())
                break;

            density_values[i%numtemp][i/numtemp]=rho;
            thermal_expansivity_values[i%numtemp][i/numtemp]=alpha;
            specific_heat_values[i%numtemp][i/numtemp]=cp;
            vp_values[i%numtemp][i/numtemp]=vp;
            vs_values[i%numtemp][i/numtemp]=vs;
            enthalpy_values[i%numtemp][i/numtemp]=h;

            i++;
        }
        AssertThrow(i==numtemp*numpress, ExcMessage("Material table size not consistent with header."
                    + Utilities::int_to_string(i)
                    + " "
                    + Utilities::int_to_string(numtemp*numpress)));

    }

    double
    specific_heat(double temperature,
                  double pressure) const
    {
        return value(temperature,pressure,specific_heat_values,interpolation);
    }

    double
    density(double temperature,
            double pressure) const
    {
        return value(temperature,pressure,density_values,interpolation);
    }

    double
    thermal_expansivity(const double temperature,
                        const double pressure) const
    {
        return value(temperature,pressure,thermal_expansivity_values,interpolation);
    }

    double
    seismic_Vp(const double temperature,
               const double pressure) const
    {
        return value(temperature,pressure,vp_values,false);
    }

    double
    seismic_Vs(const double temperature,
               const double pressure) const
    {
        return value(temperature,pressure,vs_values,false);
    }

    double
    dHdT (const double temperature,
          const double pressure) const
    {
        const double h = value(temperature,pressure,enthalpy_values,interpolation);
        const double dh = value(temperature+delta_temp,pressure,enthalpy_values,interpolation);
        return (dh - h) / delta_temp;
    }

    double
    dHdp (const double temperature,
          const double pressure) const
    {
        const double h = value(temperature,pressure,enthalpy_values,interpolation);
        const double dh = value(temperature,pressure+delta_press,enthalpy_values,interpolation);
        return (dh - h) / delta_press;
    }

    double
    dRhodp (const double temperature,
            const double pressure) const
    {
        const double rho = value(temperature,pressure,density_values,interpolation);
        const double drho = value(temperature,pressure+delta_press,density_values,interpolation);
        return (drho - rho) / delta_press;
    }

    double
    value (const double temperature,
           const double pressure,
           const dealii::Table<2,
           double> &values,
           bool interpol) const
    {
        const double nT = get_nT(temperature);
        const unsigned int inT = static_cast<unsigned int>(nT);

        const double np = get_np(pressure);
        const unsigned int inp = static_cast<unsigned int>(np);

        AssertThrow(inT<values.n_rows(), ExcMessage("Attempting to look up a temperature value with index greater than the number of rows."));
        AssertThrow(inp<values.n_cols(), ExcMessage("Attempting to look up a pressure value with index greater than the number of columns."));

        if (!interpol)
            return values[inT][inp];
        else
        {
            // compute the coordinates of this point in the
            // reference cell between the data points
            const double xi = nT-inT;
            const double eta = np-inp;

            AssertThrow ((0 <= xi) && (xi <= 1), ExcInternalError());
            AssertThrow ((0 <= eta) && (eta <= 1), ExcInternalError());

            // use these coordinates for a bilinear interpolation
            return ((1-xi)*(1-eta)*values[inT][inp] +
                    xi    *(1-eta)*values[inT+1][inp] +
                    (1-xi)*eta    *values[inT][inp+1] +
                    xi    *eta    *values[inT+1][inp+1]);
        }
    }



private:


    double get_nT(double temperature) const
    {
//        if (temperature < min_temp)
//            std::cout << "T too low " << temperature << " < " << min_temp << std::endl;
//        if (temperature > max_temp)
//            std::cout << "T too high " << temperature << " > " << max_temp << std::endl;
//            AssertThrow(temperature>=min_temp, ExcMessage("ASPECT found a temperature less than min_T: "
//                                                          + Utilities::int_to_string(int(temperature))
//                                                          + " < "
//                                                          + Utilities::int_to_string(int(min_temp))));
//            AssertThrow(temperature<=max_temp, ExcMessage("ASPECT found a temperature greater than max_T: "
//                                                          + Utilities::int_to_string(int(temperature))
//                                                          + " > "
//                                                          + Utilities::int_to_string(int(max_temp))));
        temperature=std::max(min_temp, temperature);
        temperature=std::min(temperature, max_temp-delta_temp);
        return (temperature-min_temp)/delta_temp;
    }

    double get_np(double pressure) const
    {
//        if (pressure < min_press)
//            std::cout << "P too low " << pressure << " < " << min_press << std::endl;
//        if (pressure > max_press)
//            std::cout << "P too high " << pressure << " > " << max_press << std::endl;

        pressure=std::max(min_press, pressure);
        pressure=std::min(pressure, max_press-delta_press);
        AssertThrow(pressure>=min_press, ExcMessage("ASPECT found a pressure less than min_p: "
                    + Utilities::int_to_string(int(pressure))
                    + " < "
                    + Utilities::int_to_string(int(min_press))));
        AssertThrow(pressure<=max_press, ExcMessage("ASPECT found a pressure greater than max_p: "
                    + Utilities::int_to_string(int(pressure))
                    + " > "
                    + Utilities::int_to_string(int(max_press))));
        return (pressure-min_press)/delta_press;
    }


    dealii::Table<2,double> density_values;
    dealii::Table<2,double> thermal_expansivity_values;
    dealii::Table<2,double> specific_heat_values;
    dealii::Table<2,double> vp_values;
    dealii::Table<2,double> vs_values;
    dealii::Table<2,double> enthalpy_values;


    double delta_press;
    double min_press;
    double max_press;
    double delta_temp;
    double min_temp;
    double max_temp;
    unsigned int numtemp;
    unsigned int numpress;
    bool interpolation;
};


}



template <int dim>
void
TableVp<dim>::initialize()
{

    for (unsigned i = 0; i < n_material_data; i++)
        material_lookup.push_back(std::shared_ptr<internal::MaterialLookup>
                                  (new internal::MaterialLookup(datadirectory+material_file_names[i],interpolation)));
}


template <int dim>
const std::vector<double>
TableVp<dim>::
compute_volume_fractions( const std::vector<double> &compositional_fields) const
{
    std::vector<double> volume_fractions( compositional_fields.size()+1);
    //std::vector<double> volume_fractions( compositional_fields.size());

    //clip the compositional fields so they are between zero and one
    std::vector<double> x_comp = compositional_fields;
    for ( unsigned int i=0; i < x_comp.size(); ++i)
        x_comp[i] = std::min(std::max(x_comp[i], 0.0), 1.0);

    // temporary fix for weird slabs weak zone
//    if (x_comp[0] >= 1.0 || x_comp[1] >= 1.0)
//    {
//    	x_comp[2] = std::min(0., x_comp[2]);
//    	x_comp[3] = std::min(0., x_comp[3]);
//    }

    //sum the compositional fields for normalization purposes
    double sum_composition = 0.0;
    for ( unsigned int i=0; i < x_comp.size(); ++i)
        sum_composition += x_comp[i];

    if (sum_composition >= 1.0)
    {
        volume_fractions[0] = 0.0;  //background mantle
        for ( unsigned int i=1; i <= x_comp.size(); ++i)
            volume_fractions[i] = x_comp[i-1]/sum_composition;
        /*for ( unsigned int i=0; i <= x_comp.size(); ++i)
          volume_fractions[i] = x_comp[i]/sum_composition;*/
    }
    else
    {
        volume_fractions[0] = 1.0 - sum_composition; //background mantle
        for ( unsigned int i=1; i <= x_comp.size(); ++i)
            volume_fractions[i] = x_comp[i-1];
    }
    return volume_fractions;
}


template <int dim>
double
TableVp<dim>::
viscosity (const double ,
           const double /*pressure*/,
           const std::vector<double> &,
           const SymmetricTensor<2,dim> &,
           const Point<dim> &) const
{
    return reference_eta;
}

//template <int dim>
//double
//TableVp<dim>::
//viscosity_ratio (const double temperature,
//                 const double pressure,
//                 const std::vector<double> &compositional_fields,
//                 const SymmetricTensor<2,dim> &strain_rate,
//                 const Point<dim> &position) const
//{
//    return viscosity_model->viscosity_ratio(temperature, pressure, compositional_fields, strain_rate, position);
//}



template <int dim>
double
TableVp<dim>::
get_corrected_temperature (const double temperature,
                           const double,
                           const Point<dim> &position) const
{
    if (!(this->get_adiabatic_conditions().is_initialized())
            || this->include_adiabatic_heating()
            || compressible)
        return temperature;

    return temperature
           + this->get_adiabatic_conditions().temperature(position)
           - this->get_adiabatic_surface_temperature();
}



template <int dim>
double
TableVp<dim>::
get_corrected_pressure (const double,
                        const double pressure,
                        const Point<dim> &position) const
{
    if (!(this->get_adiabatic_conditions().is_initialized())
            || compressible)
        return pressure;

    return this->get_adiabatic_conditions().pressure(position);
}

template <int dim>
double
TableVp<dim>::
get_corrected_density (const double temperature,
                       const double pressure,
                       const std::vector<double> &compositional_fields,
                       const Point<dim> &position) const
{
    const double rho = get_compressible_density(temperature,pressure,compositional_fields,position);

    const double adiabatic_temperature = this->get_adiabatic_conditions().temperature(position);
    const double adiabatic_rho = get_compressible_density(adiabatic_temperature,
                                 pressure,
                                 compositional_fields,
                                 position);

    const Point<dim> surface_point = this->get_geometry_model().representative_point(0.0);
    const double surface_temperature = this->get_adiabatic_surface_temperature();
    const double surface_pressure = this->get_surface_pressure();
    const double surface_rho = get_compressible_density(surface_temperature,
                               surface_pressure,
                               compositional_fields,
                               surface_point);

    //Return the density scaled to an incompressible profile
    const double scaled_density = (rho / adiabatic_rho) * surface_rho;
    return scaled_density;
}



template <int dim>
double
TableVp<dim>::
reference_viscosity () const
{
    return reference_eta;
}



template <int dim>
double
TableVp<dim>::
reference_density () const
{
    const double reference_density    = 3300e0;
    return reference_density;
}



template <int dim>
double
TableVp<dim>::
reference_thermal_expansion_coefficient () const
{
    return 3e-5;
}

template <int dim>
double
TableVp<dim>::
reference_specific_heat () const
{
    return 1250.0;
}

template <int dim>
double
TableVp<dim>::
reference_thermal_conductivity () const
{
    return 4.7;
}

template <int dim>
double
TableVp<dim>::
reference_thermal_diffusivity () const
{
    const double diffusivity = (reference_thermal_conductivity() / (reference_density() * reference_specific_heat()));
    return diffusivity;
}


template <int dim>
double
TableVp<dim>::
specific_heat (const double temperature,
               const double pressure,
               const std::vector<double> &compositional_fields,
               const Point<dim> &position) const
{
    double cp = 0.0;
    // test because for crust no dHdT available
    if (!latent_heat || (latent_heat && position.norm() > 6331000.0))
    {
        if (n_material_data == 1)
            cp = material_lookup[0]->specific_heat(temperature,pressure);
        else
        {
//              const unsigned int i = (unsigned int)(std::max_element( compositional_fields.begin(),
//                                                                      compositional_fields.end() )
//                                                    - compositional_fields.begin());
//              cp = material_lookup[material_per_composition[i]]->specific_heat(temperature,pressure);
            for (unsigned i = 0; i < n_fields; i++)
            {
                cp += compositional_fields[i] * material_lookup[material_per_composition[i]]->specific_heat(temperature,pressure);
            }
        }
    }
    else
    {
        if (n_material_data == 1)
            cp = material_lookup[0]->dHdT(temperature,pressure);
        else
        {
//              const unsigned int i = (unsigned int)(std::max_element( compositional_fields.begin(),
//                                                                      compositional_fields.end() )
//                                                    - compositional_fields.begin());
//              cp = material_lookup[material_per_composition[i]]->dHdT(temperature,pressure);
            for (unsigned i = 0; i < n_fields; i++)
                cp += compositional_fields[i] * material_lookup[material_per_composition[i]]->dHdT(temperature,pressure);
            cp = std::max(std::min(cp,6000.0),500.0);
        }
    }
    return cp;
}



template <int dim>
double
TableVp<dim>::
thermal_conductivity (const double,
                      const double,
                      const std::vector<double> &,
                      const Point<dim> &) const
{
    return 4.7;
}



template <int dim>
double
TableVp<dim>::
get_compressible_density (const double temperature,
                          const double pressure,
                          const std::vector<double> &compositional_fields,
                          const Point<dim> &) const
{
    double rho = 0.0;
    if (n_material_data == 1)
    {
        rho = material_lookup[0]->density(temperature,pressure);
    }
    else
    {
//          const unsigned int i = (unsigned int)(std::max_element( compositional_fields.begin(),
//                                                                  compositional_fields.end() )
//                                                - compositional_fields.begin());
//          rho = material_lookup[material_per_composition[i]]->density(temperature,pressure);
//          for (unsigned i = 0; i < n_material_data; i++)
        for (unsigned i = 0; i < n_fields; i++)
            rho += compositional_fields[i] * material_lookup[material_per_composition[i]]->density(temperature,pressure);
    }

    return rho;
}

template <int dim>
double
TableVp<dim>::
density (const double temperature,
         const double pressure,
         const std::vector<double> &compositional_fields,
         const Point<dim> &position) const
{
    if (compressible
            || !(this->get_adiabatic_conditions().is_initialized()))
        return get_compressible_density(temperature,pressure,compositional_fields,position);
    else
        return get_corrected_density(temperature,pressure,compositional_fields,position);
}



template <int dim>
double
TableVp<dim>::
thermal_expansion_coefficient (const double      temperature,
                               const double      pressure,
                               const std::vector<double> &compositional_fields,
                               const Point<dim> &position) const
{
    double alpha = 0.0;
    if (!latent_heat)
    {
        if (n_material_data == 1)
            alpha = material_lookup[0]->thermal_expansivity(temperature,pressure);
        else
        {
//              const unsigned int i = (unsigned int)(std::max_element( compositional_fields.begin(),
//                                                                      compositional_fields.end() )
//                                                    - compositional_fields.begin());
//              alpha = material_lookup[material_per_composition[i]]->thermal_expansivity(temperature,pressure);
            for (unsigned i = 0; i < n_fields; i++)
                alpha += compositional_fields[i] * material_lookup[material_per_composition[i]]->thermal_expansivity(temperature,pressure);
        }
    }
    else
    {
        double dHdp = 0.0;
        if (n_material_data == 1)
            dHdp += material_lookup[0]->dHdp(temperature,pressure);
        else
        {
//              const unsigned int i = (unsigned int)(std::max_element( compositional_fields.begin(),
//                                                                      compositional_fields.end() )
//                                                    - compositional_fields.begin());
//              dHdp = material_lookup[material_per_composition[i]]->dHdp(temperature,pressure);
            for (unsigned i = 0; i < n_fields; i++)
                dHdp += compositional_fields[i] * material_lookup[material_per_composition[i]]->dHdp(temperature,pressure);
        }
        alpha = (1 - density(temperature,pressure,compositional_fields,position) * dHdp) / temperature;
        alpha = std::max(std::min(alpha,1e-3),1e-5);
    }
    return alpha;
}



template <int dim>
double
TableVp<dim>::
seismic_Vp (const double      temperature,
            const double      pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const
{
    //this function is not called from evaluate() so we need to care about
    //corrections for temperature and pressure
    const double corrected_temperature = get_corrected_temperature(temperature,pressure,position);
    const double corrected_pressure = get_corrected_pressure(temperature,pressure,position);
    const std::vector<double> composition = compute_volume_fractions(compositional_fields);

    double vp = 0.0;
    if (n_material_data == 1)
        vp += material_lookup[0]->seismic_Vp(corrected_temperature,corrected_pressure);
    else
    {
//          const unsigned int i = (unsigned int)(std::max_element( composition.begin(),
//                                                                  composition.end() )
//                                                - composition.begin());
//          vp = material_lookup[material_per_composition[i]]->seismic_Vp(corrected_temperature,corrected_pressure);
//          ACG 05/10/2017 replace compositional_fields by composition as these functions are not called by evaluate
        for (unsigned i = 0; i < n_fields; i++)
            vp += composition[i] * material_lookup[material_per_composition[i]]->seismic_Vp(corrected_temperature,corrected_pressure);
    }
    return vp;
}



template <int dim>
double
TableVp<dim>::
seismic_Vs (const double      temperature,
            const double      pressure,
            const std::vector<double> &compositional_fields,
            const Point<dim> &position) const
{
    //this function is not called from evaluate() so we need to care about
    //corrections for temperature and pressure
    const double corrected_temperature = get_corrected_temperature(temperature,pressure,position);
    const double corrected_pressure = get_corrected_pressure(temperature,pressure,position);
    const std::vector<double> composition = compute_volume_fractions(compositional_fields);

    double vs = 0.0;
    if (n_material_data == 1)
        vs += material_lookup[0]->seismic_Vs(corrected_temperature,corrected_pressure);
    else
    {
//          const unsigned int i = (unsigned int)(std::max_element( composition.begin(),
//                                                                  composition.end() )
//                                                - composition.begin());
//          vs = material_lookup[material_per_composition[i]]->seismic_Vs(corrected_temperature,corrected_pressure);
        for (unsigned i = 0; i < n_fields; i++)
            vs += composition[i] * material_lookup[material_per_composition[i]]->seismic_Vs(corrected_temperature,corrected_pressure);
    }
    return vs;
}



template <int dim>
double
TableVp<dim>::
compressibility (const double temperature,
                 const double pressure,
                 const std::vector<double> &compositional_fields,
                 const Point<dim> &position) const
{
    if (!compressible)
        return 0.0;

    double dRhodp = 0.0;
    if (n_material_data == 1)
        dRhodp += material_lookup[0]->dRhodp(temperature,pressure);
    else
    {
//          const unsigned int i = (unsigned int)(std::max_element( compositional_fields.begin(),
//                                                                  compositional_fields.end() )
//                                                - compositional_fields.begin());
//          dRhodp = material_lookup[material_per_composition[i]]->dRhodp(temperature,pressure);
        for (unsigned i = 0; i < n_fields; i++)
            dRhodp += compositional_fields[i] * material_lookup[material_per_composition[i]]->dRhodp(temperature,pressure);
    }
    const double rho = density(temperature,pressure,compositional_fields,position);
    return (1/rho)*dRhodp;
}

template <int dim>
bool
TableVp<dim>::
is_compressible () const
{
    return compressible;
}

template <int dim>
void
TableVp<dim>::evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                       MaterialModel::MaterialModelOutputs<dim> &out) const
{

    AssertThrow ((n_material_data <= in.composition[0].size()+1) || (n_material_data == 1),
                 ExcMessage("There are more material files provided than compositional"
                            " Fields. This can not be intended."));

    MaterialModel::MaterialModelOutputs<dim> visc_out = out;

    // For compressible models,corrected temp and press = temp and press
    // TODO: only ask for viscosity?
    if (this->get_adiabatic_conditions().is_initialized() && in.strain_rate.size())
    {
        viscosity_model -> evaluate(in,visc_out);
        out.viscosities            = visc_out.viscosities;
    }

    bool in_OOC = false;
    for (unsigned int i=0; i < in.temperature.size(); ++i)
    {
      if (this->get_initial_temperature_manager().template has_matching_initial_temperature_model<const InitialTemperature::AsciiPip<dim> >())
//          (const_cast<InitialTemperature::Interface<dim> *>(&this->get_initial_temperature_manager().get_active_initial_temperature_conditions().front())))
         {
           const InitialTemperature::AsciiPip<dim> &ic = this->get_initial_temperature_manager().template get_matching_initial_temperature_model<const InitialTemperature::AsciiPip<dim> >();
           //in_OOC = ic->is_inside_2D_polygon(1,in.position[i]);
           in_OOC = ic.is_inside_2D_polygon(1,in.position[i]);
         }
        // Aegean slab
    	// If depth smaller than 120 km, use UM_pyrolite else use mantle pyrolite
      if(in.position[i].norm() > 6371000.0 - 25000.0)
           {
            if (in_OOC)
                //material_per_composition[1] = material_per_composition[5]; // change to 4 for 1 slab
              material_per_composition[n_fields-2] = material_per_composition[3];
            else
              material_per_composition[n_fields-2] = material_per_composition[1];
            }
      else if(in.position[i].norm() > 6371000.0 - 120000.0)
           //material_per_composition[1] = material_per_composition[6]; //change to 5 for 1 slab
           material_per_composition[n_fields-2] = material_per_composition[2];
      else
           //material_per_composition[1] = 0;
           material_per_composition[n_fields-2] = 0;

        // Cyprus slab
      if(in.position[i].norm() > 6371000.0 - 25000.0)
        {
         if (in_OOC)
            //material_per_composition[2] = material_per_composition[5];
            material_per_composition[n_fields-1] = material_per_composition[3];
         else
            material_per_composition[n_fields-1] = material_per_composition[1];
        }
      else 
         //material_per_composition[2] = material_per_composition[6];
         material_per_composition[n_fields-1] = material_per_composition[2];

        const double temperature = get_corrected_temperature(in.temperature[i],
                                   in.pressure[i],
                                   in.position[i]);
        const double pressure    = get_corrected_pressure(in.temperature[i],
                                   in.pressure[i],
                                   in.position[i]);

        const std::vector<double> composition = compute_volume_fractions(in.composition[i]);

        /* We are only asked to give viscosities if strain_rate.size() > 0
         * and we can only calculate it if adiabatic_conditions are available.
         * Note that the used viscosity formulation needs the not
         * corrected temperatures in case we compare it to the lateral
         * temperature average.
         */
//          if (this->get_adiabatic_conditions().is_initialized() && in.strain_rate.size())
//            {
//              out.viscosities[i]            = viscosity_model.viscosity(temperature, pressure, composition, in.strain_rate[i], in.position[i]);
//            }
        out.densities[i]                      = density                       (temperature, pressure, composition, in.position[i]);
        out.thermal_expansion_coefficients[i] = thermal_expansion_coefficient (temperature, pressure, composition, in.position[i]);
        out.specific_heat[i]                  = specific_heat                 (temperature, pressure, composition, in.position[i]);
        out.thermal_conductivities[i]         = thermal_conductivity          (temperature, pressure, composition, in.position[i]);
        out.compressibilities[i]              = compressibility               (temperature, pressure, composition, in.position[i]);
        out.entropy_derivative_pressure[i]    = 0;
        out.entropy_derivative_temperature[i] = 0;
        for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c]            = 0;
    }
}


template <int dim>
void
TableVp<dim>::declare_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("Material model");
    {
        prm.enter_subsection("Table model");
        {
            prm.declare_entry("Viscosity model","simple",
                              Patterns::Selection(MaterialModel::get_valid_model_names_pattern<dim>()),
                              "The name of a material model that will be modified by a depth "
                              "dependent viscosity. Valid values for this parameter "
                              "are the names of models that are also valid for the "
                              "``Material models/Model name'' parameter. See the documentation for "
                              "that for more information.");
            prm.declare_entry ("Data directory", "$ASPECT_SOURCE_DIR/data/material-model/steinberger/",
                               Patterns::DirectoryName (),
                               "The path to the model data. The path may also include the special "
                               "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                               "in which the ASPECT source files were located when ASPECT was "
                               "compiled. This interpretation allows, for example, to reference "
                               "files located in the 'data/' subdirectory of ASPECT. ");
            prm.declare_entry ("Material file names", "pyr-ringwood88.txt",
                               Patterns::List (Patterns::Anything()),
                               "The file names of the material data. ");
            prm.declare_entry ("Material number per composition", "0",
                               Patterns::List (Patterns::Anything()),
                               "The number of the files of the material data. "
                               "List with as many components as active "
                               "compositional fields (material data is assumed to "
                               "be in order with the ordering of the fields). ");
            prm.declare_entry ("Bilinear interpolation", "true",
                               Patterns::Bool (),
                               "Whether to use bilinear interpolation to compute "
                               "material properties (slower but more accurate). ");
            prm.declare_entry ("Latent heat", "false",
                               Patterns::Bool (),
                               "Whether to include latent heat effects in the "
                               "calculation of thermal expansivity and specific heat. "
                               "Following the approach of Nakagawa et al. 2009. ");
            prm.declare_entry ("Compressible", "false",
                               Patterns::Bool (),
                               "Whether to include a compressible material description."
                               "For a description see the manual section. ");
            prm.declare_entry ("Reference viscosity", "1e23",
                               Patterns::Double(0),
                               "The reference viscosity that is used for pressure scaling. ");
            prm.leave_subsection();
        }
        prm.leave_subsection();
    }
}



template <int dim>
void
TableVp<dim>::parse_parameters (ParameterHandler &prm)
{
    this->get_pcout() << "Really parsing material model parameters " << std::endl;
    //not pretty, but we need to get the number of compositional fields before
    //simulatoraccess has been initialized here...
    prm.enter_subsection ("Compositional fields");
    {
        n_compositional_fields = prm.get_integer ("Number of fields");
        n_fields = n_compositional_fields+1;
    }
    prm.leave_subsection();

    prm.enter_subsection("Material model");
    {
        prm.enter_subsection("Table model");
        {
            AssertThrow( prm.get("Viscosity model") != "table viscoplastic",
                         ExcMessage("You may not use ``table viscoplastic'' as the viscosity model for "
                                    "the table model.") );

            viscosity_model.reset(create_material_model<dim>(prm.get("Viscosity model")));
            if ( SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(viscosity_model.get()))
                sim->initialize_simulator (this->get_simulator());

            this->get_pcout() << "Done parsing viscosity model parameters " << std::endl;

            datadirectory        = prm.get ("Data directory");
            {
                const std::string      subst_text = "$ASPECT_SOURCE_DIR";
                std::string::size_type position;
                while (position = datadirectory.find (subst_text),  position!=std::string::npos)
                    datadirectory.replace (datadirectory.begin()+position,
                                           datadirectory.begin()+position+subst_text.size(),
                                           subst_text);
            }

            this->get_pcout() << "Done finding data directory " << std::endl;
            material_file_names  = Utilities::split_string_list
                                   (prm.get ("Material file names"));

            n_material_data = material_file_names.size();
            this->get_pcout() << "    Number of materials found: " << n_material_data << std::endl;

            const std::vector<int> materials = Utilities::string_to_int(Utilities::split_string_list(prm.get ("Material number per composition")));
            AssertThrow(materials.size() == n_fields, ExcMessage("Please specify a material table number for each composition. "));
            material_per_composition.resize(n_fields);
            for (unsigned int i = 0; i < n_fields; i++)
            {
                AssertThrow((unsigned int)(materials[i]) < n_material_data, ExcMessage("Please specify a valid material data number."));
                material_per_composition[i] = (unsigned int)(materials[i]);
            }

            interpolation        = prm.get_bool ("Bilinear interpolation");
            latent_heat          = prm.get_bool ("Latent heat");
            compressible         = prm.get_bool ("Compressible");
            reference_eta        = prm.get_double ("Reference viscosity");

            prm.leave_subsection();
        }
        prm.leave_subsection();

        this->get_pcout() << "Setting model dependencies " << std::endl;
        // Declare dependencies on solution variables
        this->model_dependence.viscosity = NonlinearDependence::temperature;
        this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
        this->model_dependence.compressibility = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
        this->model_dependence.specific_heat = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
        this->model_dependence.thermal_conductivity = NonlinearDependence::none;
    }
    /* After parsing the parameters for depth dependent, it is essential to parse
    parameters related to the base model. */
    this->get_pcout() << "Parsing viscosity model " << std::endl;
    viscosity_model->parse_parameters(prm);
    this-> model_dependence.viscosity = viscosity_model->get_model_dependence().viscosity;

    this->get_pcout() << "Done parsing table model parameters " << std::endl;
}
}
}


// explicit instantiations
namespace aspect
{
namespace MaterialModel
{
ASPECT_REGISTER_MATERIAL_MODEL(TableVp,
                               "table viscoplastic",
                               "This material model looks up the viscosity from the tables that "
                               "correspond to the paper of Steinberger and Calderwood "
                               "2006 (``Models of large-scale viscous flow in the Earth's "
                               "mantle with constraints from mineral physics and surface observations'', "
                               "Geophys. J. Int., 167, 1461-1481, "
                               "\\url{http://dx.doi.org/10.1111/j.1365-246X.2006.03131.x}) and material "
                               "data from a database generated by the thermodynamics code \\texttt{Perplex}, "
                               "see \\url{http://www.perplex.ethz.ch/}. "
                               "The default example data builds upon the thermodynamic "
                               "database by Stixrude 2011 and assumes a pyrolitic composition by "
                               "Ringwood 1988 but is easily replaceable by other data files. ")
}
}
