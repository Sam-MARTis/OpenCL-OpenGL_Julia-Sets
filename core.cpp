#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <string>
#include <iomanip>
#include <cstdint>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 300
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
#include <CL/cl.h>
#include <CL/cl_gl.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glx.h>
#include <GL/gl.h>

#define NX 256*4
#define NY 256*4
#define DNX 50
#define DNY 50
#define LEFT -1.5
#define RIGHT 1.5
#define UP 2.0
#define DOWN -2.0
#define ITERATIONS_COUNT 500
#define ZOOM_FACTOR 0.95
#define MOVEMENT_DELTA 0.1
#define C_DELTA 0.01

#define CX 0.0
#define CY 0.7

#define WORKSIZE_Y 32
#define WORKSIZE_X 32


#define KERNEL_FILEPATH "render_program.cl"
#define KERNEL_FUNC "find_convergence"
#define OUT_FILENAME "out_gpu.ppm"

int win;
GLuint tex;
cl_command_queue cmdq;
cl_kernel kernel;
cl_mem clImage;
cl_program program;
cl_context ctx;
cl_device_id dev;
cl_platform_id platform;

double xstart;
double ystart;
double cx = CX;
double cy = CY;
int iterations;

const int countx = NX;
const int county = NY;

double right = RIGHT; 
double left = LEFT;
double up = UP;
double down = DOWN;
double dx = (RIGHT - LEFT) / ((double)NX);
double dy = (UP - DOWN) / ((double)NY);

double delta = MOVEMENT_DELTA;
double c_delta = C_DELTA;




cl_platform_id getPlatform()
{
    cl_platform_id platform;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Unable to get platform");
        exit(1);
    }
    return platform;
}

cl_device_id getDevice(cl_platform_id &platform)
{
    cl_device_id dev;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Unable to get platform");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        std::cout << "Unable to get GPU. trying CPU\n";
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
        if (err < 0)
        {
            perror("Unable to get any device\n");
            exit(1);
        }
    }

    cl_device_fp_config cfg;
    err = clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
    if (err != CL_SUCCESS || cfg == 0) {
    printf("Double precision not supported\n");
    exit(1);
} else {
    printf("Double precision supported!\n");
}
    return dev;
}

cl_program getProgram(cl_context &ctx, cl_device_id &dev, const char *filepath)
{
    FILE *program_handle;
    char *program_buffer;
    size_t program_size;
    cl_program program;
    cl_int err;
    program_handle = fopen(filepath, "r");
    if (program_handle == NULL)
    {
        perror("Unable to open file\n");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc((program_size + 1) * sizeof(char));
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("unable to create program from source\n");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {
        char *program_log;
        size_t log_size;
        perror("Unable to build program:\n\n");
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc((log_size + 1) * sizeof(char));
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        free(program_log);
    }

    return program;
}
cl_context getContext(cl_platform_id &platform, cl_device_id &dev)
{
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        0};

    cl_int err;
    cl_context ctx = clCreateContext(props, 1, &dev, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Unable to create context\n");
        exit(1);
    }
    return ctx;
}

void setKernelArgs(cl_kernel &kernel, cl_mem &_clImage, const double _left, const double _down, const int _countx, const int _county, const double _dx, const double _dy, const double _cx, const double _cy, const int _iterations)
{
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &_clImage);
    if (err < 0)
    {
        std::cerr << "Error setting arg 0: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 1, sizeof(double), &_left);
    if (err < 0)
    {
        std::cerr << "Error setting arg 1: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 2, sizeof(double), &_down);
    if (err < 0)
    {
        std::cerr << "Error setting arg 2: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 3, sizeof(int), &_countx);
    if (err < 0)
    {
        std::cerr << "Error setting arg 3: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 4, sizeof(int), &_county);
    if (err < 0)
    {
        std::cerr << "Error setting arg 4: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 5, sizeof(double), &_dx);
    if (err < 0)
    {
        std::cerr << "Error setting arg 5: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 6, sizeof(double), &_dy);
    if (err < 0)
    {
        std::cerr << "Error setting arg 6: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 7, sizeof(double), &_cx);
    if (err < 0)
    {
        std::cerr << "Error setting arg 7: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 8, sizeof(double), &_cy);
    if (err < 0)
    {
        std::cerr << "Error setting arg 8: " << err << std::endl;
    }
    err |= clSetKernelArg(kernel, 9, sizeof(int), &_iterations);
    if (err < 0)
    {
        std::cerr << "Error setting arg 9: " << err << std::endl;
    }

    if (err < 0)
    {
        perror("Unable to set a kernel arguement");
        exit(1);
    }
}

void launchKernel(cl_kernel &kernel, cl_command_queue &cmdq)
{
    cl_int err;
    const int workgroup_size_x = WORKSIZE_X;
    const int workgroup_size_y = WORKSIZE_Y;
    const int workgroups_x = (int)ceil(((double)NX) / ((double)WORKSIZE_X));
    const int workgroups_y = (int)ceil(((double)NY) / ((double)WORKSIZE_Y));
    const int global_size_x = workgroup_size_x * workgroups_x;
    const int global_size_y = workgroup_size_y * workgroups_y;

    size_t global_size[2] = {global_size_x, global_size_y};
    size_t local_size[2] = {workgroup_size_x, workgroup_size_y};

    err = clEnqueueNDRangeKernel(cmdq, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    if (err < 0)
    {
        perror("Couldn't enqueue the kernel");
        exit(1);
    }
}

GLuint createTexture()
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, NX, NY, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    return tex;
}

cl_mem createCLImage(GLuint &tex, cl_context &ctx)
{
    cl_int err;
    cl_mem cl_image = clCreateFromGLTexture(ctx, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex, &err);
    if (err < 0)
    {
        perror("Unable go create image from texture\n");
        exit(1);
    }
    return cl_image;
}

void initOpenGLGlut(int argc, char **argv)
{

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(NX, NY);
    glutInitWindowPosition(DNX, DNY);
    win = glutCreateWindow("OpenCL + OpenGL inteop test: Julia sets");
    printf("Window id: %d\n", win);
}

cl_command_queue createCommandQueue(cl_context &ctx, cl_device_id &dev)
{
    cl_int err;
    cl_command_queue cmdq = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    if (err < 0)
    {
        perror("Unable to create command queue");
        exit(1);
    }
    return cmdq;
}


void performComputation()
{
    clEnqueueAcquireGLObjects(cmdq, 1, &clImage, 0, NULL, NULL);
    clFinish(cmdq);
    setKernelArgs(kernel, clImage, left, down, countx, county, dx, dy, cx, cy, iterations);
    launchKernel(kernel, cmdq);
    clFinish(cmdq);
    clEnqueueReleaseGLObjects(cmdq, 1, &clImage, 0, NULL, NULL);
    clFinish(cmdq);
}
void renderFunction()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, NX, NY, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, NY);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(NX, NY);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(NX, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();
    glutPostRedisplay();
}
void keyboardBindings(unsigned char key, int x, int y){

    if( key=='w') {
        up += delta;
        down += delta;
    }
    if( key=='s') {
        up -= delta;
        down -= delta;
    }
    if( key=='a') {
        left -= delta;
        right -= delta;
    }
    if( key=='d') {
        left += delta;
        right += delta;
    }
    if(key=='+'){
        double xRange = right - left;
        double yRange = up - down;
        double xCenter = (right + left) / 2.0f;
        double yCenter = (up + down) / 2.0f;
        xRange *= ZOOM_FACTOR;
        yRange *= ZOOM_FACTOR;
        left = xCenter - xRange / 2.0f;
        right = xCenter + xRange / 2.0f;
        up = yCenter + yRange / 2.0f;
        down = yCenter - yRange / 2.0f;
        dx = (right - left) / ((double)NX);
        dy = (up - down) / ((double)NY);
        delta *= ZOOM_FACTOR;
    }
    if(key=='-'){
        double xRange = right - left;
        double yRange = up - down;
        double xCenter = (right + left) / 2.0f;
        double yCenter = (up + down) / 2.0f;

        xRange /= ZOOM_FACTOR;
        yRange /= ZOOM_FACTOR;
        left = xCenter - xRange / 2.0f;
        right = xCenter + xRange / 2.0f;
        up = yCenter + yRange / 2.0f;
        down = yCenter - yRange / 2.0f;
        dx = (right - left) / ((double)NX);
        dy = (up - down) / ((double)NY);
        delta /= ZOOM_FACTOR;

    }
    
    if(key == 't')
    {
        cy += c_delta;
  
    }
    
    if (key == 'g')
    {
        cy -= c_delta;
  
    }
    
    if (key == 'f')
    {
        cx -= c_delta;

    }
    
    if (key == 'h')
    {
        cx += c_delta;

    }
    performComputation();
    glutPostRedisplay();


}


int main(int argc, char **argv)
{
    initOpenGLGlut(argc, argv);
    GLenum res = glewInit();

    if (res != GLEW_OK)
    {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return 1;
    }
    cl_int err;
    platform = getPlatform();
    dev = getDevice(platform);
    ctx = getContext(platform, dev);
    program = getProgram(ctx, dev, KERNEL_FILEPATH);
    cmdq = createCommandQueue(ctx, dev);
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0)
    {
        perror("Unable to create kernel");
        exit(1);
    }

    tex = createTexture();
    GLenum glErr = glGetError();
    clImage = createCLImage(tex, ctx);
    std::cout << "clImage: " << clImage << std::endl;
    if (clImage == NULL)
    {
        perror("Unable to create cl image");
        exit(1);

    }

    iterations = ITERATIONS_COUNT;
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glViewport(0, 0, NX, NY);
    gluOrtho2D(0, NX, 0, NY);

    glutDisplayFunc(renderFunction);
    glutKeyboardFunc(keyboardBindings);
    performComputation();
    glutMainLoop();

    clReleaseKernel(kernel);
    clReleaseMemObject(clImage);
    clReleaseCommandQueue(cmdq);
    clReleaseProgram(program);
    clReleaseContext(ctx);
    return 0;
}
