/*
  Copyright (C) 2011 - 2019 by the authors of the ASPECT code.

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


#include <aspect/postprocess/source_host_fault_statistics.h>
#include <aspect/material_model/simple.h>
#include <aspect/global.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    SourceHostFaultStatistics<dim>::execute (TableHandler &statistics)
    {
      if (this->n_compositional_fields() == 0)
        return std::pair<std::string,std::string>();

      // create a quadrature formula based on the compositional element alone.
      // be defensive about determining that a compositional field actually exists
      AssertThrow (this->introspection().base_elements.compositional_fields
                   != numbers::invalid_unsigned_int,
                   ExcMessage("This postprocessor cannot be used without compositional fields."));

      // Check that the required fields exist
      AssertThrow (this->introspection().compositional_name_exists("sediment"), ExcMessage("This postprocessor can only be "
                   "used if a field called sediment is present."));
      AssertThrow (this->introspection().compositional_name_exists("silt_fraction"), ExcMessage("This postprocessor can only be "
                   "used if a field called silt_fraction is present."));
      AssertThrow (this->introspection().compositional_name_exists("ratio_marine_continental_sediment"), 
                   ExcMessage("This postprocessor can only be "
                   "used if a field called ratio_marine_continental_sediment is present."));

      // Retrieve the field number of the relevant fields.
      const unsigned int sediment_field = this->introspection().compositional_index_for_name("sediment");
      const unsigned int marine_fraction_field = this->introspection().compositional_index_for_name("ratio_marine_continental_sediment");
      const unsigned int silt_fraction_field = this->introspection().compositional_index_for_name("silt_fraction");

      // Assume temperature and composition use the same degree 
      const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.compositional_fields).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_gradients   |
                               update_quadrature_points |
                               update_JxW_values);

      std::vector<SymmetricTensor<2,dim>> strain_rate_values(n_q_points);
      std::vector<double> temperature_values(n_q_points);
      std::vector<double> sediment_values(n_q_points);
      std::vector<double> silt_fraction_values(n_q_points);
      std::vector<double> marine_fraction_values(n_q_points);

      double local_source_integral = 0;
      double local_host_integral = 0;
      double local_fault_integral = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            // Get strain rate
            fe_values[this->introspection().extractors.velocities]
                  .get_function_symmetric_gradients (this->get_solution(), strain_rate_values);
            // Get temperature
            fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
                                                                                        temperature_values);
            // Get compositions
            fe_values[this->introspection().extractors.compositional_fields[sediment_field]].get_function_values (this->get_solution(),
                    sediment_values);
            fe_values[this->introspection().extractors.compositional_fields[silt_fraction_field]].get_function_values (this->get_solution(),
                    silt_fraction_values);
            fe_values[this->introspection().extractors.compositional_fields[marine_fraction_field]].get_function_values (this->get_solution(),
                    marine_fraction_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
               bool source_present = false;
               bool host_present = false;
               // Sand with T > 250 C
               if (sediment_values[q] >= 0.5 &&
                   marine_fraction_values[q] < 0.5 &&
                   silt_fraction_values[q] < 0.5 &&
                   temperature_values[q] >= minimum_source_temperature)
                {
                 local_source_integral += fe_values.JxW(q);
                 source_present = true;
                }

                // Pelagic sediments with T < 150 C
               if (sediment_values[q] >= 0.5 &&
                   marine_fraction_values[q] >= 0.5 &&
                   temperature_values[q] < maximum_host_temperature)
                {
                 local_host_integral += fe_values.JxW(q);
                 host_present = true;
                }
 
               if ((source_present || host_present) &&
                   std::sqrt(std::fabs(second_invariant(deviator(strain_rate_values[q])))) >= strain_rate_threshold)
                 local_fault_integral += fe_values.JxW(q);

              }
          }

      const double global_source_integral
        = Utilities::MPI::sum (local_source_integral, this->get_mpi_communicator());
      const double global_host_integral
        = Utilities::MPI::sum (local_host_integral, this->get_mpi_communicator());
      const double global_fault_integral
        = Utilities::MPI::sum (local_fault_integral, this->get_mpi_communicator());

      const std::string units = (dim == 2) ? "m^2" : "m^3";
      const std::vector<std::string> column_names = {"Source rock (" + units + ")",
                                                     "Host rock (" + units + ")",
                                                     "Fault rock (" + units + ")"
                                                    };

      statistics.add_value (column_names[0],
                            global_source_integral);
      statistics.add_value (column_names[1],
                            global_host_integral);
      statistics.add_value (column_names[2],
                            global_fault_integral);

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      for (auto &column : column_names)
        {
          statistics.set_precision (column, 8);
          statistics.set_scientific (column, true);
        }

      std::ostringstream output;
      output.precision(4);
      output << global_source_integral
             << ' ' << units << ", "
             << global_host_integral
             << ' ' << units << ", "
             << global_fault_integral
             << ' ' << units;

      return std::pair<std::string, std::string> ("Source rock, host rock, fault rock:",
                                                  output.str());
    }



    template <int dim>
    void
    SourceHostFaultStatistics<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Source and host rock statistics");
        {
          prm.declare_entry ("Fault strain rate threshold", "1e-15",
                             Patterns::Double (0.),
                             "The scalar effective strain rate value that acts "
                             "as the threshold to define a fault. "
                             "Units: 1/s. ");
          prm.declare_entry ("Source rock minimum temperature", "250",
                             Patterns::Double (0.),
                             "The minimum temperature that the source rock should have. "
                             "Units: C. ");
          prm.declare_entry ("Host rock maximum temperature", "150",
                             Patterns::Double (0.),
                             "The maximum temperature that the host rock should have. "
                             "Units: C. ");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    SourceHostFaultStatistics<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Source and host rock statistics");
        {
          strain_rate_threshold = prm.get_double ("Fault strain rate threshold");
          minimum_source_temperature = prm.get_double ("Source rock minimum temperature") + 273.25;
          maximum_host_temperature = prm.get_double ("Host rock maximum temperature") + 273.25;
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
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(SourceHostFaultStatistics,
                                  "source host fault statistics",
                                  "A postprocessor that computes some statistics about the "
                                  "area/volume of potential source rock and host rock for ore deposits.")
  }
}
