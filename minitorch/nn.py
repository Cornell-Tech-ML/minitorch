from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators

# from .tensor import Tensor


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError('Need to implement for Task 4.3')
    # return input.view(batch, channel, int(height / kh), int(width / kw), kh * kw), int(height / kh), int(width / kw)

    # out = input.contiguous().view(batch, channel, int(height / kh), int(width / kw), kh * kw)
    # for i in range(batch):
    #     for j in range(channel):
    #         for h in range(int(height / kh)):
    #             for w in range(int(width / kw)):
    #                 counter = 0
    #                 for hh in range(kh):
    #                     for ww in range(kw):
    #                         out[i, j, h, w, counter] = input[i, j, h * kh + hh, w  * kw + ww]
    #                         counter += 1
    # return out, int(height / kh), int(width / kw)
    new_height = int(height / kh)
    new_width = int(width / kw)
    return (
        input.contiguous()
        .view(batch, channel, height, new_width, kw)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_height, new_width, int(kh * kw)),
        new_height,
        new_width,
    )


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError('Need to implement for Task 4.3')
    out, new_height, new_width = tile(input, kernel)
    # return out.view(batch, channel, new_height, new_width)
    out = out.mean(4)
    return out.view(*out.shape[:-1])


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    input = input + rand(input.shape) * 1e-5
    out = max_reduce(input, [dim])
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError('Need to implement for Task 4.4')
        ctx.save_for_backward(input, dim)
        return max_reduce(input, [dim])

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError('Need to implement for Task 4.4')
        input, dim = ctx.saved_values
        # out = max_reduce(input, [dim])
        # return (out == input) * grad_output
        return argmax(input, dim) * grad_output


max = Max.apply

add_reduce = FastOps.reduce(operators.add)


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    # add_reduce = FastOps.reduce(operators.add)
    # input = input + rand(input.shape) * 1e-5
    exp_input = input.exp()
    exp_input_deno = exp_input.sum(dim=dim)  # add_reduce(exp_input, [dim])
    return exp_input / exp_input_deno
    # e_x = (input - max(input, dim))
    # return e_x / e_x.sum(dim)


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    exp_input = input.exp()
    exp_input_deno = exp_input.sum(dim=dim).log()  # add_reduce(exp_input, [dim]).log()
    return input - exp_input_deno


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    out, new_height, new_width = tile(input, kernel)
    # return out.view(batch, channel, new_height, new_width)
    out = max(out, 4)
    return out.view(*out.shape[:-1])


# relu_map = FastOps.map(operators.relu)
lt_zip = FastOps.zip(operators.lt)


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    r = rand(input.shape)
    p = input.zeros() + 1 * rate
    return lt_zip(p, r) * input
