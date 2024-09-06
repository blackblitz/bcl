"""Extended module for neural-tangents."""

from typing import Optional

from jax import eval_shape, linear_transpose, jvp, vjp, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp

from .empirical import (
    _add, _diagonal, _get_args, _get_f1_f2, _ndim,
    _std_basis, _trace_and_diagonal, _unravel_array_into_pytree
)
from .utils import utils
from .utils.typing import ApplyFn, Axes, EmpiricalKernelFn, PyTree, VMapAxes


def empirical_ntk(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> EmpiricalKernelFn:
    """Compute NTK via NTK-vector products."""

    def ntk_fn(
        x1: PyTree,
        x2: Optional[PyTree],
        params: PyTree,
        sigma: PyTree,
        **apply_fn_kwargs
    ) -> jnp.ndarray:
        """Compute a sample of the empirical NTK with NTK-vector products.

        Args:
            x1:
                first batch of inputs.

            x2:
                second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must
                have a matching shape with `f(x1)` on `trace_axes` and
                `diagonal_axes`.

            params:
                A `PyTree` of parameters about which we would like to compute
                the neural tangent kernel.

            sigma:
                A `PyTree` of parameters that will transform

            **apply_fn_kwargs:
                keyword arguments passed to `apply_fn`. `apply_fn_kwargs`
                will be split into `apply_fn_kwargs1` and `apply_fn_kwargs2`
                by the `split_kwargs` function which will be passed to
                `apply_fn`. In particular, the rng key in `apply_fn_kwargs`,
                will be split into two different (if `x1 != x2`) or same
                (if `x1 == x2`) rng keys. See the `_read_key` function for
                more details.

        Returns:
            A single sample of the empirical NTK. The shape of the kernel is
            "almost" `zip(f(x1).shape, f(x2).shape)` except for:
            1) `trace_axes` are absent as they are contracted over.
            2) `diagonal_axes` are present only once.
            All other axes are present twice.
        """
        args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
                f, apply_fn_kwargs, params, vmap_axes, x1, x2)

        def get_ntk(x1, x2, *args):
            f1, f2 = _get_f1_f2(
                f, keys, x_axis, fx_axis, kw_axes, args, x1, x2
            )

            def delta_vjp_jvp(delta):
                def delta_vjp(delta):
                    return vjp(f2, params)[1](delta)
                return jvp(
                    f1, (params,),
                    (tree_map(
                        lambda x, y: x * y,
                        sigma, delta_vjp(delta)[0]
                    ),)
                )[1]

            fx1, fx2 = eval_shape(f1, params), eval_shape(f2, params)
            eye = _std_basis(fx1)
            ntk = vmap(linear_transpose(delta_vjp_jvp, fx2))(eye)
            ntk = tree_map(
                lambda fx12: _unravel_array_into_pytree(fx1, 0, fx12), ntk
            )
            ntk = _diagonal(ntk, fx1)
            return ntk

        if not utils.all_none(x_axis) or not utils.all_none(kw_axes):
            x2 = x1 if utils.all_none(x2) else x2

            kw_in_axes = [kw_axes[k] if k in kw_axes else None for k in keys]
            in_axes1 = [x_axis, None] + kw_in_axes + [None] * len(kw_in_axes)
            in_axes2 = [None, x_axis] + [None] * len(kw_in_axes) + kw_in_axes

            get_ntk = vmap(
                vmap(get_ntk, in_axes1, fx_axis),
                in_axes2, _add(fx_axis, _ndim(fx1))
            )

        ntk = get_ntk(x1, x2, *args1, *args2)
        ntk = tree_map(
            lambda x: _trace_and_diagonal(x, trace_axes, diagonal_axes), ntk
        )
        return ntk

    return ntk_fn
