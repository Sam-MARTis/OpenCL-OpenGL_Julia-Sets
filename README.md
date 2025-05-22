## Julia Sets

The basic mathematic idea is to check the convergence of the sequence:
$$Z_{n+1} = Z_n^2 + C\ \ \ \ \ \ \ C, Z_i \in \mathbb{C}\ \ \ \ \  \forall i \in \mathbb{N}$$
for various values of $Z_0$. This sequence also results in the well known Mandelbrot set.

However in Julia sets, we vary $C$ and plot teh resultant set. $Z_0$ is chosen as the cooridnate of each pixel and its corresponding convergence value is computed in the GPU kernel in parallel for each pixel.

$C$ can be varied using keybindings:
t => Im(C) += Delta
g => Im(C) -= Delta
f => Re(C) -= Delta
h => Re(C) += Delta

Additionally, for movement of the frame:
w, a, s, d => up, left, down, right.
\+, \- => Zoom in, Zoom out


### Requirements for running this sim
(This is primarily designed for **my** laptop. I don't really it worth trying to make it cross compatible)
- Linux. There are additional properties that need to be set when creating the context for windows. OpenCL is not supported on Apple.
I dont have the willpower to do it but if you can figure it out, here's where u need to change it:
```cpp
cl_context getContext(cl_platform_id &platform, cl_device_id &dev)
{
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        0};
    // Other code. The above props need to be changed. glX is for OSX, window has different. Be sure to change the glx header file as well. Top of the file.
}
```
- GPU must support double precision. Just run the program and it will output if your gpu supports double precision.
- OpenCL version 3.0 (You *probably* already have this)
- OpenGL latest(atleast not outdated) version.

### How to run
If you have two gpus, make sure to use the dedicated gpu. Or else OpenCL and OpenGL will create contexts in different GPUs.
For NVIDIA you can do it by typing the following in your console before running the executable to force OpenGL to use the dedicated gpu:
```sh
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```


To run just do:
```sh
./julia
```
