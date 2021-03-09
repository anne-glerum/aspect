/*
  Copyright (C) 2014 - 2020 by the authors of the ASPECT code.

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


#ifndef _aspect_time_stepping_lapusta_time_step_h
#define _aspect_time_stepping_lapusta_time_step_h

#include <aspect/time_stepping/interface.h>
#include <aspect/material_model/rheology/friction_options.h>
#include <aspect/material_model/rheology/visco_plastic.h>


namespace aspect
{
  namespace TimeStepping
  {
    using namespace dealii;

    /**
     * Compute the lapusta time step based on the current solution and
     * return it.
     *
     * @ingroup TimeStepping
     */
    template <int dim>
    class LapustaTimeStep : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        LapustaTimeStep () = default;


        /**
         * @copydoc aspect::TimeStepping::Interface<dim>::execute()
         */
        virtual
        double
        execute() override;

        /**
         * Object for computing the friction dependence of the rheology model.
         */
        MaterialModel::Rheology::FrictionOptions<dim> friction_options;

        /**
         * Object for computing the friction dependence of the rheology model.
         */
        MaterialModel::Rheology::ViscoPlastic<dim> visco_plastic;

      private:
    };
  }
}


#endif
