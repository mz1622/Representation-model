"""Standalone frozen numeric basis prototype; not connected to any R8 model."""
import torch
from torch import nn


class FrozenNumericBasis(nn.Module):
    """Append bounded per-axis ramps while retaining every visible raw scalar.

    Knots must already come from a separately fingerprinted training-only fit.
    Output layout is [masked values, visibility, flattened axis-major ramps].
    Exact duplicates are removed in float32 deployment space. Empty/constant
    axes use zero ramps, retaining their raw scalar and visibility channels.
    There are no trainable parameters and no fitting in forward or load.
    """

    def __init__(self, knots, max_intervals):
        super().__init__()
        if type(max_intervals) is not int or max_intervals < 1:
            raise ValueError('max_intervals must be a positive integer.')
        knots = list(knots)
        if not knots:
            raise ValueError('At least one axis is required.')
        self.axes = len(knots)
        self.max_intervals = max_intervals
        self.output_features = self.axes * (2 + max_intervals)
        lower = torch.zeros(self.axes, max_intervals)
        width = torch.ones_like(lower)
        active = torch.zeros_like(lower, dtype=torch.bool)
        for axis, values in enumerate(knots):
            values = torch.as_tensor(values, dtype=torch.float32, device='cpu').detach()
            if values.ndim != 1 or not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError('Knots must be finite nonnegative one-dimensional sequences.')
            if len(values) > 1 and (values[1:] < values[:-1]).any():
                raise ValueError('Knots must be sorted; no implicit reordering.')
            values = torch.unique_consecutive(values)
            count = max(0, len(values) - 1)
            if count > max_intervals:
                raise ValueError('More effective intervals than allocated slots.')
            if count:
                lower[axis, :count] = values[:-1]
                width[axis, :count] = values[1:] - values[:-1]
                active[axis, :count] = True
        self.register_buffer('lower', lower)
        self.register_buffer('width', width)
        self.register_buffer('active', active)
        self.validate()

    def validate(self):
        shape = (self.axes, self.max_intervals)
        if self.lower.shape != shape or self.width.shape != shape or self.active.shape != shape:
            raise ValueError('Saved numeric basis shape mismatch.')
        if self.active.dtype != torch.bool:
            raise ValueError('Interval visibility must be boolean.')
        if not torch.isfinite(self.lower).all() or (self.lower < 0).any():
            raise ValueError('Invalid lower knots.')
        if not torch.isfinite(self.width).all() or (self.width <= 0).any():
            raise ValueError('Invalid interval widths; duplicate knots must be removed.')

    def forward(self, values, visible):
        if values.ndim != 2 or values.shape[1] != self.axes or visible.shape != values.shape:
            raise ValueError('Expected matching batch-by-axis value and visibility matrices.')
        if visible.dtype != torch.bool or values.dtype not in (torch.float32, torch.float64):
            raise ValueError('Boolean visibility and float32/float64 values are required.')
        if values.device != self.lower.device or visible.device != values.device or values.dtype != self.lower.dtype:
            raise ValueError('Values, visibility and basis must share device and floating dtype.')
        self.validate()
        # Hidden payloads, including NaN placeholders, never reach feature construction.
        safe = torch.where(visible, values, 0.)
        if not torch.isfinite(safe).all() or (safe < 0).any():
            raise FloatingPointError('Visible values must be finite nonnegative scaled-log inputs.')
        # Cap the numerator before division: even tiny positive widths cannot
        # create an intermediate infinity for a finite out-of-range value.
        numerator = (safe.unsqueeze(-1) - self.lower).clamp_min(0.)
        ramps = torch.minimum(numerator, self.width) / self.width
        ramps = torch.where(visible.unsqueeze(-1) & self.active, ramps, 0.)
        result = torch.cat([safe, visible.to(values.dtype), ramps.flatten(1)], dim=1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite numeric basis output.')
        return result
