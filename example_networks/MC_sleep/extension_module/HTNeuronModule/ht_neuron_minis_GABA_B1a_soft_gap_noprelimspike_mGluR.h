/*
 *  ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR_H
#define ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR_H

// Generated includes:
#include "config.h"

#ifdef HAVE_GSL

// C++ includes:
#include <string>
#include <vector>

// C includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

// Includes from sli:
#include "stringdatum.h"
#include "normal_randomdev.h"

/* BeginDocumentation
   Name: ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR - Neuron model after Hill & Tononi (2005).
   Description:
   This model neuron implements a slightly modified version of the
   neuron model described in [1]. The most important properties are:
   - Integrate-and-fire with threshold adaptive threshold.
   - Repolarizing potassium current instead of hard reset.
   - AMPA, NMDA, GABA_A, and GABA_B conductance-based synapses with
     beta-function (difference of exponentials) time course.
   - Voltage-dependent NMDA with instantaneous or two-stage unblocking [1, 2].
   - Intrinsic currents I_h, I_T, I_Na(p), and I_KNa.
   - Synaptic "minis" are not implemented.
   Documentation and Examples:
   - docs/model_details/HillTononiModels.ipynb
   - pynest/examples/intrinsic_currents_spiking.py
   - pynest/examples/intrinsic_currents_subthreshold.py
   Parameters:
   V_m            - membrane potential
   tau_m          - membrane time constant applying to all currents except
                    repolarizing K-current (see [1], p 1677)
   t_ref          - refractory time and duration of post-spike repolarizing
                    potassium current (t_spike in [1])
   tau_spike      - membrane time constant for post-spike repolarizing
                    potassium current
   voltage_clamp  - if true, clamp voltage to value at beginning of simulation
                    (default: false, mainly for testing)
   theta, theta_eq, tau_theta - threshold, equilibrium value, time constant
   g_KL, E_K, g_NaL, E_Na     - conductances and reversal potentials for K and
                                Na leak currents
   {E_rev,g_peak,tau_rise,tau_decay}_{AMPA,NMDA,GABA_A,GABA_B}
                                - reversal potentials, peak conductances and
                                  time constants for synapses (tau_rise/
                                  tau_decay correspond to tau_1/tau_2 in the
                                  paper)
   V_act_NMDA, S_act_NMDA, tau_Mg_{fast, slow}_NMDA
                                - parameters for voltage dependence of NMDA-
                                  conductance, see above
   instant_unblock_NMDA         - instantaneous NMDA unblocking (default: false)
   {E_rev,g_peak}_{h,T,NaP,KNa} - reversal potential and peak conductance for
                                  intrinsic currents
   tau_D_KNa                    - relaxation time constant for I_KNa
   receptor_types               - dictionary mapping synapse names to ports on
                                  neuron model
   recordables                  - list of recordable quantities
   equilibrate                  - if given and true, time-dependent activation
                                  and inactivation state variables (h, m) of
                                  intrinsic currents and NMDA channels are set
                                  to their equilibrium values during this
                                  SetStatus call; otherwise they retain their
                                  present values.
   Note: Conductances are unitless in this model and currents are in mV.
   Author: Hans Ekkehard Plesser
   Sends: SpikeEvent
   Receives: SpikeEvent, CurrentEvent, DataLoggingRequest
   FirstVersion: October 2009; full revision November 2016
   References:
   [1] S Hill and G Tononi (2005). J Neurophysiol 93:1671-1698.
   [2] M Vargas-Caballero HPC Robinson (2003). J Neurophysiol 89:27782783.
   SeeAlso: ht_synapse
*/

namespace mynest
{
/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL. Internally, it is
 *       a first-class C++ function, but cannot be a member function
 *       because of the C-linkage.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 * @param void* Pointer to model neuron instance.
 */
extern "C" int ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR_dynamics( double, const double*, double*, void* );

class ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR : public nest::Archiving_Node
{
public:
  ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR();
  ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR( const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR& );
  ~ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR();

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using nest::Node::handle;
  using nest::Node::handles_test_event;
  using nest::Node::sends_secondary_event;

  nest::port send_test_event( nest::Node&, nest::rport, nest::synindex, bool );

  void handle( nest::SpikeEvent& e );
  void handle( nest::CurrentEvent& e );
  void handle( nest::DataLoggingRequest& );
  void handle( nest::GapJunctionEvent& );

  nest::port handles_test_event( nest::SpikeEvent&, nest::rport );
  nest::port handles_test_event( nest::CurrentEvent&, nest::rport );
  nest::port handles_test_event( nest::DataLoggingRequest&, nest::rport );
  nest::port handles_test_event( nest::GapJunctionEvent&, nest::rport );

  void
  sends_secondary_event( nest::GapJunctionEvent& )
  {
  }

  /**
   * Return membrane potential at time t.
potentials_.connect_logging_device();
   * This function is not thread-safe and should not be used in threaded
   * contexts to access the current membrane potential values.
   * @param Time the current network time
   *
   */
  double get_potential( nest::Time const& ) const;

  /**
   * Define current membrane potential.
   * This function is thread-safe and should be used in threaded
   * contexts to change the current membrane potential value.
   * @param Time     the current network time
   * @param double new value of the mebrane potential
   *
   */
  void set_potential( nest::Time const&, double );

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  /**
   * Synapse types to connect to
   * @note Excluded upper and lower bounds are defined as INF_, SUP_.
   *       Excluding port 0 avoids accidental connections.
   */
  enum SynapseTypes
  {
    INF_SPIKE_RECEPTOR = 0,
    AMPA,
    NMDA,
    GABA_A,
    GABA_B,
    mGluR,
    GABA_B1a, // Keep after all other receptors and before MINI
    MINI, // Keep last (not scaled by GABA_B1a)
    SUP_SPIKE_RECEPTOR
  };

  void init_state_( const Node& proto );
  void init_buffers_();
  void calibrate();

  //! Take neuron through given time interval
  /** This is the actual update function. The additional boolean parameter
   * determines if the function is called by update (false) or wfr_update (true)
   */
  bool update_( nest::Time const&, const long, const long, const bool );

  void update( nest::Time const&, const long, const long );
  bool wfr_update( nest::Time const&, const long, const long );

  double get_synapse_constant( double, double, double );

  // END Boilerplate function declarations ----------------------------

  // Friends --------------------------------------------------------

  // make dynamics function quasi-member
  friend int ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR_dynamics( double, const double*, double*, void* );

  // ----------------------------------------------------------------

  /**
   * Independent parameters of the model.
   */
  struct Parameters_
  {
    Parameters_();

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
    void set( const DictionaryDatum& ); //!< Set values from dicitonary

    // Note: Conductances are unitless
    // Leaks
    double E_Na; // mV
    double E_K;  // mV
    double g_NaL;
    double g_KL;
    double tau_m; // ms

    // Dynamic threshold
    double theta_eq;  // mV
    double tau_theta; // ms

    // Post-spike potassium current
    double tau_spike; // ms, membrane time constant for this current
    double t_ref;     // ms, refractory time

    // Parameters for synapse of type AMPA, GABA_A, GABA_B, GABA_B1a and NMDA
    double g_peak_AMPA;
    double tau_rise_AMPA;  // ms
    double tau_decay_AMPA; // ms
    double E_rev_AMPA;     // mV

    double g_peak_NMDA;
    double tau_rise_NMDA;  // ms
    double tau_decay_NMDA; // ms
    double E_rev_NMDA;     // mV
    double V_act_NMDA;     // mV, inactive for V << Vact, inflection of sigmoid
    double S_act_NMDA;     // mV, scale of inactivation
    double tau_Mg_slow_NMDA; // ms
    double tau_Mg_fast_NMDA; // ms
    bool instant_unblock_NMDA;

    double g_peak_GABA_A;
    double tau_rise_GABA_A;  // ms
    double tau_decay_GABA_A; // ms
    double E_rev_GABA_A;     // mV

    double g_peak_GABA_B;
    double tau_rise_GABA_B;  // ms
    double tau_decay_GABA_B; // ms
    double E_rev_GABA_B;     // mV

    double g_peak_mGluR;
    double tau_rise_mGluR;  // ms
    double tau_decay_mGluR; // ms
    double E_rev_mGluR;     // mV

    // GABA_B1a: Downscaling of incoming inputs
    // Incoming spikes are scaled down by w -> w * x_GABA_B1a
    // Where x_GABA_B1a = max(0,
    //                         min(xmax_GABA_B1a,
    //                             1 - alpha * min(g_max_GABA_B1a, g_GABA_B1a)))
    // g_GABA_B1a is a double exponential synaptic conductance which is soft
    // bound (and hard bound) by g_max_GABA_B1a: all incoming GABA_B1a spikes
    // have an effective g_peak of g_peak_GABA_B1a * (1 -
    // g_GABA_B1a/g_max_GABA_B1a)
    double xmax_GABA_B1a;      //!< Maximum scaling weight. Between 0 and 1
    double g_peak_GABA_B1a;     //!< Peak GABA_B1a unitary conductance
    double g_max_GABA_B1a;      //!< Soft and hard upper bound for GABA_B1a
                                  // conductance. Greater than 0
    double tau_rise_GABA_B1a; //!< Rise time constant for GABA_B1a currents
    double tau_decay_GABA_B1a; //!< Decay time constant for GABA_B1a currents
    /* Scaling factor: one value between 0.0 and 1.0 for each receptor in the
       neuron (AMPA, NMDA, GABA_A, GABA_B, mGluR, GABA_B1a) */
    std::vector< double > alpha_GABA_B1a;

    // Parameters for synapse of type MINI
    double mini_mean;
    double mini_sigma;

    // Parameters for gap junction
    double g_gap_scale_factor; // (unitless) Scaling factor for I_gap (~conductance)

    // parameters for intrinsic currents
    double g_peak_NaP;
    double E_rev_NaP; // mV

    double g_peak_KNa;
    double E_rev_KNa; // mV
    double tau_D_KNa; // ms

    double g_peak_T;
    double E_rev_T; // mV

    double g_peak_h;
    double E_rev_h; // mV

    bool voltage_clamp;
  };

  // ----------------------------------------------------------------

  /**
   * State variables of the model.
   */
public:
  struct State_
  {

    // y_ = [V, theta, Synapses]
    enum StateVecElems_
    {
      V_M = 0,
      THETA,
      DG_AMPA,
      G_AMPA,
      DG_NMDA_TIMECOURSE,
      G_NMDA_TIMECOURSE,
      DG_GABA_A,
      G_GABA_A,
      DG_GABA_B,
      G_GABA_B,
      DG_mGluR,
      G_mGluR,
      DG_GABA_B1a,
      G_GABA_B1a,
      // DO NOT INSERT ANYTHING UP TO HERE, WILL MIX UP
      // SPIKE DELIVERY
      m_fast_NMDA,
      m_slow_NMDA,
      m_Ih,
      D_IKNa,
      m_IT,
      h_IT,
      STATE_VEC_SIZE
    };

    //! neuron state, must be C-array for GSL solver
    double y_[ STATE_VEC_SIZE ];

    /** Timer (counter) for spike-activated repolarizing potassium current.
     * Neuron is absolutely refractory during this period.
     */
    long ref_steps_;

    double I_NaP_; //!< Persistent Na current; member only to allow recording
    double I_KNa_; //!< Depol act. K current; member only to allow recording
    double I_T_;   //!< Low-thresh Ca current; member only to allow recording
    double I_h_;   //!< Pacemaker current; member only to allow recording
    double I_gap_;

    State_( const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR&, const Parameters_& p );
    State_( const State_& s );
    ~State_();

    State_& operator=( const State_& s );

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum&, const ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR& );
  };

private:
  // These friend declarations must be precisely here.
  friend class nest::RecordablesMap< ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR >;
  friend class nest::UniversalDataLogger< ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR >;


  // ----------------------------------------------------------------

  /**
   * Buffers of the model.
   */
  struct Buffers_
  {
    Buffers_( ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR& );
    Buffers_( const Buffers_&, ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR& );

    nest::UniversalDataLogger< ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR > logger_;

    /** buffers and sums up incoming spikes/currents */
    std::vector< nest::RingBuffer > spike_inputs_;
    nest::RingBuffer currents_;

    /** GSL ODE stuff */
    gsl_odeiv_step* s_;    //!< stepping function
    gsl_odeiv_control* c_; //!< adaptive stepsize control function
    gsl_odeiv_evolve* e_;  //!< evolution function
    gsl_odeiv_system sys_; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double step_;             //!< step size in ms
    double integration_step_; //!< current integration time step, updated by GSL

    /**
     * Input current injected by CurrentEvent.
     * This variable is used to transport the current applied into the
     * _dynamics function computing the derivative of the state vector.
     * It must be a part of Buffers_, since it is initialized once before
     * the first simulation, but not modified before later Simulate calls.
     */
    double I_stim_;

    // remembers current lag for piecewise interpolation
    long lag_;
    // remembers y_values from last wfr_update
    std::vector< double > last_y_values;
    // summarized gap weight
    double sumj_g_ij_;
    // summarized coefficients of the interpolation polynomial
    std::vector< double > interpolation_coefficients;
  };

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {
    //! size of conductance steps for arriving spikes
    std::vector< double > cond_steps_;

    //! Duration of potassium current.
    int PotassiumRefractoryCounts_;

    //! Voltage at beginning of simulation, for clamping
    double V_clamp_;

    //! Random variable for minis
    librandom::NormalRandomDev normal_dev_;
  };


  // readout functions, can use template for vector elements
  template < State_::StateVecElems_ elem >
  double
  get_y_elem_() const
  {
    return S_.y_[ elem ];
  }
  double
  get_I_NaP_() const
  {
    return S_.I_NaP_;
  }
  double
  get_I_KNa_() const
  {
    return S_.I_KNa_;
  }
  double
  get_I_T_() const
  {
    return S_.I_T_;
  }
  double
  get_I_h_() const
  {
    return S_.I_h_;
  }

  double get_g_NMDA_() const;

  double
  get_I_gap_() const
  {
    return S_.I_gap_;
  }

  double get_x_GABA_B1a( int i_receptor ) const;

  // For recording
  double get_x_GABA_B1a() const
  {
      int recorded_i = 1;
      return get_x_GABA_B1a( recorded_i );
  }

  double get_g_GABA_B1a() const
  {
      return std::min(
        P_.g_max_GABA_B1a,
        S_.y_[ State_::G_GABA_B1a ]
      );
  }
  /**
   * NMDA activation for given parameters
   * Needs to take parameter values explicitly since it is called from
   * _dynamics.
   */
  double m_NMDA_( double V, double m_eq, double m_fast, double m_slow ) const;

  /**
   * Return equilibrium value of I_h activation
   *
   * @param V Membrane potential for which to evaluate
   *        (may differ from y_[V_M] when clamping)
   */
  double m_eq_h_( double V ) const;

  /**
   * Return equilibrium value of I_T activation
   *
   * @param V Membrane potential for which to evaluate
   *        (may differ from y_[V_M] when clamping)
   */
  double m_eq_T_( double V ) const;

  /**
   * Return equilibrium value of I_T inactivation
   *
   * @param V Membrane potential for which to evaluate
   *        (may differ from y_[V_M] when clamping)
   */
  double h_eq_T_( double V ) const;

  /**
   * Return steady-state magnesium unblock ratio.
   *
   * Receives V_m as argument since it is called from ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR_dyamics
   * with temporary state values.
   */
  double m_eq_NMDA_( double V ) const;

  /**
   * Steady-state "D" value for given voltage.
   */
  double D_eq_KNa_( double V ) const;

  static nest::RecordablesMap< ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR > recordablesMap_;

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;
};

inline void
ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::update( nest::Time const& origin, const long from,
    const long to )
{
  update_( origin, from, to, false );
}

inline bool
ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::wfr_update( nest::Time const& origin,
  const long from,
  const long to )
{
  State_ old_state = S_; // save state before wfr_update
  const bool wfr_tol_exceeded = update_( origin, from, to, true );
  S_ = old_state; // restore old state

  return not wfr_tol_exceeded;
}

inline nest::port
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::send_test_event( Node& target,
        nest::rport receptor_type,
        nest::synindex,
        bool )
{
  nest::SpikeEvent e;
  e.set_sender( *this );

  return target.handles_test_event( e, receptor_type );
}


inline nest::port
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::handles_test_event( nest::SpikeEvent&, nest::rport receptor_type )
{
  assert( B_.spike_inputs_.size() == SUP_SPIKE_RECEPTOR-INF_SPIKE_RECEPTOR-1 );

  if ( !( INF_SPIKE_RECEPTOR < receptor_type
         && receptor_type < SUP_SPIKE_RECEPTOR ) )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
    return 0;
  }
  else
    return receptor_type - 1;


  /*
if (receptor_type != 0)
throw UnknownReceptorType(receptor_type, get_name());
return 0;*/
}

inline nest::port
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::handles_test_event( nest::CurrentEvent&, nest::port receptor_type )
{

  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline nest::port
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline nest::port
mynest::ht_neuron_minis_GABA_B1a_soft_gap_noprelimspike_mGluR::handles_test_event( nest::GapJunctionEvent&, nest::rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

}

#endif // HAVE_GSL
#endif // HT_NEURON_H
