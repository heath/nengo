"""
simulator_objects.py: model description classes

These classes are used to describe a Nengo model to be simulated.
Model is the input to a *simulator* (see e.g. simulator.py).

"""
import numpy as np

from .nonlinear import registry as nlreg


random_weight_rng = np.random.RandomState(12345)

class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""


class SignalView(object):
    """Interpretable, vector-valued quantity within NEF
    """
    def __init__(self, base, shape, elemstrides, offset):
        self.base = base
        self.shape = tuple(shape)
        self.elemstrides = tuple(elemstrides)
        self.offset = int(offset)

    def __len__(self):
        return self.shape[0]

    @property
    def dtype(self):
        return self.base._dtype

    @property
    def size(self):
        return int(np.prod(self.shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self.elemstrides == (1,):
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        else:
            # -- there are cases where reshaping can still work
            #    but there are limits too, because we can only
            #    support view-based reshapes. So the strides have
            #    to work.
            raise TODO('reshape of strided view')

    def transpose(self, neworder=None):
        raise TODO('transpose')

    def __getitem__(self, item):
        # -- copy the shape and strides
        shape = list(self.shape)
        elemstrides = list(self.elemstrides)
        offset = self.offset
        if isinstance(item, (list, tuple)):
            dims_to_del = []
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * elemstrides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * elemstrides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                elemstrides.pop(dim)
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, (int, np.integer)):
            if len(self.shape) == 0:
                raise IndexError()
            if not (0 <= item < self.shape[0]):
                raise NotImplementedError()
            shape = self.shape[1:]
            elemstrides = self.elemstrides[1:]
            offset = self.offset + item * self.elemstrides[0]
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, slice):
            return self.__getitem__((item,))
        else:
            raise NotImplementedError(item)


class Signal(SignalView):
    """Interpretable, vector-valued quantity within NEF"""
    def __init__(self, n=1, dtype=np.float64):
        self.n = n
        self._dtype = dtype

    @property
    def shape(self):
        return (self.n,)

    @property
    def elemstrides(self):
        return (1,)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self


class SignalProbe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, n, value):
        Signal.__init__(self, n)
        self.value = value


class CustomComputation(object):
    """A custom computation. Implements the same interface as populations."""
    def __init__(self, func):
        self.func = func


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        self.alpha = alpha
        self.insig = insig
        self.outsig = outsig


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        self.oldsig = oldsig
        self.newsig = newsig
        self.alpha = alpha

    def __str__(self):
        return '%s{%s, %s, %s}' % (
            self.__class__.__name__,
            self.alpha, self.oldsig, self.newsig)

    def __repr__(self):
        return str(self)


class Encoder(object):
    """A linear transform from a signal to a population"""
    def __init__(self, sig, pop, weights=None):
        self.sig = sig
        self.pop = pop
        if weights is None:
            weights = random_weight_rng.randn(pop.n, sig.size)
        else:
            weights = np.asarray(weights)
            if weights.shape != (pop.n, sig.size):
                raise ValueError('weight shape', weights.shape)
        self.weights = weights


class Decoder(object):
    """A linear transform from a population to a signal"""
    def __init__(self, pop, sig, weights=None):
        self.pop = pop
        self.sig = sig
        if weights is None:
            weights = random_weight_rng.randn(sig.size, pop.n)
        else:
            weights = np.asarray(weights)
            if weights.shape != (sig.size, pop.n):
                raise ValueError('weight shape', weights.shape)
        self.weights = weights


class SimModel(object):
    """
    A container for model components.
    """
    def __init__(self, dt=0.001):
        self.dt = dt
        self.signals = []
        self.nonlinearities = []
        self.encoders = []
        self.decoders = []
        self.transforms = []
        self.filters = []
        self.signal_probes = []

    def signal(self, n=1, value=None):
        """Add a signal to the model"""
        if value is None:
            rval = Signal(n)
        else:
            rval = Constant(n, value)
        self.signals.append(rval)
        return rval

    def signal_probe(self, sig, dt):
        """Add a signal probe to the model"""
        rval = SignalProbe(sig, dt)
        self.signal_probes.append(rval)
        return rval

    def nonlinearity(self, nlclass, *args, **kwargs):
        """Add a nonlinearity (some computation) to the model"""
        if not nlclass in nlreg:
            raise TypeError('The "' + nl.__class__.__name__ + '" nonlinearity '
                            'is not in nengo.nonlinear.registry.')
        rval = nlclass(*args, **kwargs)
        self.nonlinearities.append(rval)
        return rvla

    def encoder(self, sig, pop, weights=None):
        """Add an encoder to the model"""
        rval = Encoder(sig, pop, weights=weights)
        self.encoders.append(rval)
        return rval

    def decoder(self, pop, sig, weights=None):
        """Add a decoder to the model"""
        rval = Decoder(pop, sig, weights=weights)
        self.decoders.append(rval)
        return rval

    def transform(self, alpha, insig, outsig):
        """Add a transform to the model"""
        rval = Transform(alpha, insig, outsig)
        self.transforms.append(rval)
        return rval

    def filter(self, alpha, oldsig, newsig):
        """Add a filter to the model"""
        rval = Filter(alpha, oldsig, newsig)
        self.filters.append(rval)
        return rval