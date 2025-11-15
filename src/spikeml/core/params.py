from typing import Any, Optional
from pydantic import BaseModel, Field, WithJsonSchema

from spikeml.utils.dict_util import dic2str
from spikeml.utils.vector import UpsampleMethod

# =============================================================================
# Global Default Parameters
# =============================================================================

# Feedback
G = 1      # Output gain
E_ERR = 5  # Error exponent

# Mean-rate neuron parameters
VMIN = 0   # Lower bound on activation (mean firing rate)
VMAX = 1   # Upper bound on activation (mean firing rate)

# Stochastic spikes
PF = 1     # Spike probability factor
PMAX = 1   # Maximum spike probability cutoff
E_Z = 2    # Spike probability exponent

# Adaptive threshold
E_B = 2    # Exponent for adaptive threshold
T_B = 20   # Characteristic time for adaptive threshold

# Long-Term Potentiation/Depression (LTP/LTD)
CMAX = 2   # Maximum synaptic connection weight
CMIN = -2  # Minimum synaptic connection weight
T_P = 1    # Characteristic time for potentiation
T_D = 5    # Characteristic time for depression
T_C = 10   # Characteristic time for connection decay
C_K = 2    # LTP/LTD threshold
C_IN = 0   # Connection normalization (inbound)
C_OUT = 0  # Connection normalization (outbound)

# Leaky Integrate-and-Fire neuron
T_X = 20   # Leak time constant
K_X = 1    # Firing threshold
R_SD = 0.01  # Random noise standard deviation

class Params(BaseModel):
    """
    Base parameter model for neural network components.

    Provides string formatting and convenience methods for displaying
    model parameters in a human-readable way.
    """
        
    def __str__(self):
        """Return a formatted string representation of the parameter set."""
        
        s = str(vars(self))
        return f'{type(self).__name__}({s})'        

    def fmt(self, sep=','):
        """
        Return parameters as a formatted key-value string.

        Parameters
        ----------
        sep : str, optional
            Separator between key-value pairs. Default is ','.

        Returns
        -------
        str
            Formatted representation of parameters.
        """
        return f'{type(self).__name__}:{dic2str(vars(self), sep)}'        

# =============================================================================
# Parameter Models
# =============================================================================


class NNParams(Params):
    """
    Base parameters for neural network units.

    Attributes
    ----------
    g : float
        Output gain (scaling factor).
    e_err : float
        Exponent used for error transformation.
    vmin : float
        Minimum neuron activation value (mean rate lower bound).
    vmax : float
        Maximum neuron activation value (mean rate upper bound).
    name : str, optional
        Optional name of the neural unit or layer.
    """
    g: float = Field(default=G, gt=0)
    e_err: float = Field(default=E_ERR)
    vmin: float = Field(default=VMIN, ge=0)
    vmax: float = Field(default=VMAX, ge=0)
    name: Optional[str] = Field(default=None)
    #xx: Annotated[str, Field(default=None, strict=False), WithJsonSchema({'extra': 'data'})]
    
    def __str__(self):
        s = str(vars(self))
        return f'{type(self).__name__}({s})'        

class ConnectorParams(Params):
    """
    Synaptic connection parameters governing LTP/LTD and normalization.

    Attributes
    ----------
    c_k : float
        LTP/LTD activation threshold.
    t_c : float
        Time constant for connection decay.
    t_p : float
        Time constant for LTP (potentiation).
    t_d : float
        Time constant for LTD (depression).
    c_in : float
        Inbound connection normalization coefficient.
    c_out : float
        Outbound connection normalization coefficient.
    cmin : float
        Minimum allowed synaptic weight.
    cmax : float
        Maximum allowed synaptic weight.
    mean : float
        Mean of initial connection distribution.
    sd : float
        Standard deviation of initial connection distribution.
    size : Any
        Shape or size of the connection matrix.
    """
    c_k: float = Field(default=C_K)
    t_c: float = Field(default=T_C, ge=0)
    t_p: float = Field(default=T_P, ge=0)
    t_d: float = Field(default=T_D, ge=0)
    c_in: float = Field(default=C_IN)
    c_out: float = Field(default=C_OUT)
    cmin: float = Field(default=CMIN)
    cmax: float = Field(default=CMAX)
    mean: float = Field(default=0)
    sd: float = Field(default=.1, ge=0)
    size: Any = Field(default=None)

class SpikeParams(NNParams):
    """
    Parameters controlling stochastic spike generation.

    Attributes
    ----------
    e_z : float
        Exponent controlling the spike probability nonlinearity.
    pf : float
        Spike probability scaling factor.
    pmax : float
        Maximum spike probability cutoff (0 ≤ pmax ≤ 1).
    """    
    e_z: float = Field(default=E_Z)
    pf: float = Field(default=PF)
    pmax: float = Field(default=PMAX, ge=0, le=1)
    
    def __str__(self):
        s = str(vars(self))
        return f'{type(self).__name__}({s})'        

class SSensorParams(SpikeParams):
    """
    Parameters for standard Spiking Neural Network (SNN) models.

    Combines spiking dynamics with synaptic plasticity and leaky integration.

    Attributes
    ----------
    ssize : int
        Number of sensor units
    upsample_method: UpsampleMethod
        upsampling method. Defualt: UUpsampleMethodsamplingMethod.REPEAT'
    """    
    upsample_method: UpsampleMethod = Field(default=UpsampleMethod.REPEAT)
    
    pass

class SNNParams(SpikeParams, ConnectorParams): #TODO: -> NNParams
    """
    Parameters for standard leaky-integrate-fire Spiking Neural Network (SNN) models.

    Combines spiking dynamics with synaptic plasticity and leaky integration.

    Attributes
    ----------
    t_x : float
        Leak time constant for the neuron membrane potential.
    k_x : float
        Membrane firing threshold.
    r_sd : float
        Standard deviation of random noise in membrane potential.
    """  
    t_x: float = Field(default=T_X, ge=0)
    k_x: float = Field(default=K_X, ge=0)
    r_sd: float = Field(default=R_SD, ge=0)

    
class SSNNParams(SpikeParams, ConnectorParams):
    """
    Parameters for stochastic adaptive Spiking Neural Network (SNN) models.

    Sthochatic leaky-integrate-fire SNNs with adaptive threshold mechanisms for enhanced
    temporal stability and homeostasis.

    Attributes
    ----------
    e_z : float
        Exponent controlling spike probability.
    pf : float
        Spike probability scaling factor.
    pmax : float
        Maximum spike probability cutoff.
    t_b : float
        Characteristic time for adaptive threshold dynamics.
    e_b : float
        Exponent controlling adaptive threshold scaling.
    """    
    e_z: float = Field(default=E_Z)
    pf: float = Field(default=PF)
    pmax: float = Field(default=PMAX, ge=0, le=1)
    t_b: float = Field(default=T_B, ge=0)
    e_b: float = Field(default=E_B, ge=0)

