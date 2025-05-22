#define MAX_RES 0.6

__kernel
void find_convergence(write_only image2d_t output,  const double xstart,  const double ystart,  const int countx, const int county,  const double dx, const double dy,  const double cx,  const double cy, const int iterations){
    size_t idx = get_global_id(0);
    size_t idy = get_global_id(1);
    if((idx>=countx)||(idy>=county)) return;
    double x = xstart + dx*(double)idx;
    double y = ystart + dy*(double)idy;
    int iterations_completed = 0;
    while((iterations_completed<iterations) && ((x*x + y*y)<4)){
        const double x_temp = x;
        x = x*x - y*y + cx;
        y = 2*x_temp*y + cy;
        iterations_completed++;
    }

    double rsq = x*x + y*y;
    double res = MAX_RES*((double)(iterations_completed))/((double)iterations);

    res = res>MAX_RES ? MAX_RES : (res<0.0 ? 0.0 : res);
    double r = (double)(res);
    double g = (double)(pow(res, 3));
    double b = (double)(pow(res, 2));
    float4 colour = (float4)(r, g, b, 1.0f);
    int2 coord = (int2)(idx, idy);
    write_imagef(output, coord, colour);
}

