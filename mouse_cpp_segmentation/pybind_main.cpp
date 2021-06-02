#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <armadillo>
#include <math.h>
#include "iostream"
#include <pybind11/numpy.h>

//#include <pyNiftiImage.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;
int add(int i, int j) {
    return i + j;
}


void tm2(py::array_t<int>& a){
    auto buf  = a.request();
    int* pt = (int*) buf.ptr;
    pt[0] = 10;

}

void test_method(){
    py::array_t<int> a = py::array_t<int>(2);
    auto buf  = a.request();

    int* pt = (int*) buf.ptr;

    pt[0] = 1;
    pt[1] = 4;

    tm2(a);
    std::cout << pt[0] <<std::endl;

}


double c1_calc(py::array_t<double>& image
        ,py::array_t<int>& u)

{
    py::buffer_info buf1 = u.request();
    py::buffer_info buf2 = image.request();

    long x_last_ind = buf1.shape[0] -1;
    long y_last_ind = buf1.shape[1] -1;
    long z_last_ind = buf1.shape[2] -1;

    long X = buf1.shape[0];
    long Y = buf1.shape[1];
    long Z = buf1.shape[2];

    double* buf_image = (double *) buf2.ptr;
    int* buf_u = (int *) buf1.ptr;

    double den=10e-8;
    double nom = 0;

    for (int i = 0;i< X;i++){
        for (int j = 0;j< Y;j++){
            for (int k = 0;k< Z;k++){
                nom += buf_image[i*Y*Z + j*Z + k]*(buf_u[i*Y*Z + j*Z + k]);
                den += (buf_u[i*Y*Z + j*Z + k]);
            }
        }
    }

    return nom/den;

}

double c0_calc(py::array_t<double>& image
        ,py::array_t<int> u)

{
    py::buffer_info buf1 = u.request();
    py::buffer_info buf2 = image.request();

    long x_last_ind = buf1.shape[0] -1;
    long y_last_ind = buf1.shape[1] -1;
    long z_last_ind = buf1.shape[2] -1;

    long X = buf1.shape[0];
    long Y = buf1.shape[1];
    long Z = buf1.shape[2];

    double* buf_image = (double *) buf2.ptr;
    int* buf_u = (int *) buf1.ptr;

    double den=10e-8;
    double nom = 0;

    for (int i = 0;i< X;i++){
        for (int j = 0;j< Y;j++){
            for (int k = 0;k< Z;k++){
                nom += buf_image[i*Y*Z + j*Z + k]*(1- buf_u[i*Y*Z + j*Z + k]);
                den += (1-buf_u[i*Y*Z + j*Z + k]);
            }
        }
    }

    return nom/den;

}

py::array_t<double> abs_grad(py::array_t<int> u){

    py::buffer_info buf1 = u.request();

    double axis_spacing_x = 1;
    double axis_spacing_y = 1;
    double axis_spacing_z = 1;

    long x_last_ind = buf1.shape[0] -1;
    long y_last_ind = buf1.shape[1] -1;
    long z_last_ind = buf1.shape[2] -1;

    long X = buf1.shape[0];
    long Y = buf1.shape[1];
    long Z = buf1.shape[2];
    //Z axis gradient

    py::array_t<double> result = py::array_t<double>(buf1.size);
    py::buffer_info buf3 = result.request();
    double xd,yd,zd;
    int *ptr1 = (int *) buf1.ptr;

    double *resarr = (double *) buf3.ptr;


    for (int i = 0;i< X;i++){

        for (int j = 0;j< Y;j++){



            for (int k = 0;k< Z;k++){

                //std::cout << "c6666" << std::endl;
                if (i==0){
                    xd = ( (ptr1[(i+1)*Y*Z + j*Z + k] - ptr1[i*Y*Z + j*Z + k])/axis_spacing_x  );
                }
                if (j==0){
                    yd =( (ptr1[i*Y*Z + (j+1)*Z + k] - ptr1[i*Y*Z + j*Z + k])/axis_spacing_y  );
                }
                if (k==0){
                    zd =( (ptr1[i*Y*Z + j*Z + k+1] - ptr1[i*Y*Z + j*Z + k])/axis_spacing_z  );
                }
                if (i!=0 && i!=(x_last_ind) ){
                    xd =( double (ptr1[(i+1)*Y*Z + j*Z + k] - ptr1[(i-1)*Y*Z + j*Z + k])/2*axis_spacing_x  );
                }
                if (j!=0 && j!=(y_last_ind) ){
                    yd =( double (ptr1[i*Y*Z + (j+1)*Z + k] - ptr1[i*Y*Z + (j-1)*Z + k])/2*axis_spacing_y  );
                }
                if (k!=0 && k!=(z_last_ind) ){
                    zd =( double (ptr1[i*Y*Z + j*Z + k+1] - ptr1[i*Y*Z + j*Z + k-1])/2*axis_spacing_z  );
                }

                if (i==(x_last_ind) ){
                    xd =( double (ptr1[i*Y*Z + j*Z + k] - ptr1[(i-1)*Y*Z + j*Z + k])/axis_spacing_x  );
                }
                if (j==(y_last_ind) ){
                    yd =( double (ptr1[i*Y*Z + j*Z + k] - ptr1[i*Y*Z + (j-1)*Z + k])/axis_spacing_y  );
                }
                if (k==(z_last_ind)){
                    zd =( double (ptr1[i*Y*Z + j*Z + k] - ptr1[i*Y*Z + j*Z + k-1])/axis_spacing_z  );
                }

                resarr[i*Y*Z + j*Z + k] = (abs(xd) + abs(yd) + abs(zd) );

            }

        }
    }
    result.resize({X,Y,Z});
    return result;
}


void modify_u_mat(py::array_t<double>& image,
                  py::array_t<double>& abs_du
        ,py::array_t<int>& u, //output
        double lambda1,double lambda2,
        double c0,double c1){

    py::buffer_info buf1 = u.request();

    py::buffer_info buf_absdu = abs_du.request();
    py::buffer_info buf_image = image.request();


    double axis_spacing_x = 1;
    double axis_spacing_y = 1;
    double axis_spacing_z = 1;

    int x_last_ind = buf1.shape[0] -1;
    int y_last_ind = buf1.shape[1] -1;
    int z_last_ind = buf1.shape[2] -1;

    int X = buf1.shape[0];
    int Y = buf1.shape[1];
    int Z = buf1.shape[2];

    double *abs_ptr =(double *)buf_absdu.ptr;
    double *img_ptr =(double *)buf_image.ptr;
    int *u_ptr =(int *)buf1.ptr;
    double tmp;
    for (int i = 0;i< X;i++){
        for (int j = 0;j< Y;j++){
            for (int k = 0;k<Z;k++){
                //std::cout << "i " << i << " j " << j << " k " << k<< std::endl;
                tmp = abs_ptr[i*Y*Z + j*Z + k] * (lambda1*(img_ptr[i*Y*Z + j*Z + k] - c1)*(img_ptr[i*Y*Z + j*Z + k] - c1) -
                        lambda2*(img_ptr[i*Y*Z + j*Z + k] - c0)*(img_ptr[i*Y*Z + j*Z + k] - c0)
                        );
                if (tmp < 0) u_ptr[i*Y*Z + j*Z + k]=1;
                if (tmp > 0) u_ptr[i*Y*Z + j*Z + k]=0;

            }
        }
    }
    std::cout << "EXIT" << std::endl;
    u.resize({X,Y,Z});


}



py::array_t<int> morph_cv(py::array_t<double> image
                                                    ,py::array_t<int> init_level_set,
                                                    int iterations,
        int smoothing=1,double lambda1=1,double lambda2=1)
{
    py::module_ skim_sg = py::module_::import("skimage.segmentation");
    //skim_sg.
    auto u = init_level_set;

    for (int l1 =0; l1 < iterations;l1++){

        double c0 = c0_calc(image,u);
        //std::cout << "c0" << std::endl;
        double c1 = c1_calc(image,u);
        //std::cout << "c1" << std::endl;
        auto abs_gr = abs_grad(u);
        //std::cout << "AG" << std::endl;
        modify_u_mat(image,abs_gr,u,lambda1,lambda2,c0,c1);
        std::cout << "AG2" << std::endl;
//        for (int l2 =0; l2 < iterations;l2++){
//
//
//
//
//        }



    }
    return u;


}

py::array_t<double> ret_arrays(py::array_t<double> input1){

    py::buffer_info buf1 = input1.request();
    if (buf1.ndim != 3)
        throw std::runtime_error("Number of dimensions must be one");
    int X = buf1.shape[0];
    int Y = buf1.shape[1];
    int Z = buf1.shape[2];

    double *ptr1 = (double *) buf1.ptr;
    for (size_t idx = 0; idx < X; idx++) {
        for (size_t idy = 0; idy < Y; idy++) {
            for (size_t idz = 0; idz < Z; idz++) {
                std::cout << ptr1[idx*Y*Z+ idy*Z + idz] << std::endl;
            }
        }
    }
    return input1;
}




using namespace pybind11::literals;


PYBIND11_MODULE(Mouse_C, m) {
m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

m.def("add", py::vectorize(add));

//m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
//        Subtract two numbers
//        Some other explanation about the subtract function.
//    )pbdoc");
//
    m.def("ret_array", &ret_arrays, R"pbdoc(
        C++ impl of morphological CV

    )pbdoc");
    m.def("test_method", &test_method, R"pbdoc(
        C++ impl of morphological CV

    )pbdoc");

    m.def("morph_cv", &morph_cv,"image"_a,"init_level_set"_a,"iterations"_a,"smoothing"_a=1,"lambda1"_a=1
            ,"lambda2"_a=1, R"pbdoc(
        C++ impl of morphological CV

    )pbdoc");



//m.def( "is_triangle_intersected",&pySurface::triangles_intersected,R"pbdoc(
//        Test if triangles formed by points intersected
//        Some other explanation about the subtract function.
//    )pbdoc");
//
//    py::class_<pySurface>(m, "cMesh")
//            .def(py::init<const std::string &>())
//            .def("getName",&pySurface::getName)
//            .def("modify_points", &pySurface::modify_points)
//            .def("selfIntersectionTest", &pySurface::self_intersection_test)
//            .def("apply_transform", &pySurface::apply_transformation)
//            .def("generate_normals", &pySurface::generateNormals)
//            .def("get_faces",&pySurface::getFaces)
//            .def("save_obj",&pySurface::saveObj)
//            .def("generate_mesh_points", &pySurface::getInsideMeshPoints)
//            .def("get_unpacked_coords",&pySurface::getUnpackedCords)
//            .def("calculate_volume",&pySurface::computeVolume)
//            .def("get_mesh_boundary_roi", &pySurface::getInsideBoundaryPoints)
//            .def("is_points_inside",&pySurface::isPointsInside)
//            .def("ray_mesh_intersection", &pySurface::rayTriangleIntersection)
//            .def( "centes_of_triangles", &pySurface::centresOfTriangles)
//            .def( "index_of_intersectedtriangle", &pySurface::rayTriangleIntersectionIndexes);


//    py::class_<pyNiftiImage>(m,"cImage")
//            .def(py::init<std::string>())
//            .def("loadMask", &pyNiftiImage::setMask)
//            .def("interpolate_normals", &pyNiftiImage::interpolate_normals);

#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif
}
