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

double m_c1_calc(py::array_t<double>& image
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
                nom += buf_image[k*Y*X+j*X+i]*(buf_u[i*Y*Z+j*Z+k]);
                den += (buf_u[i*Y*Z+j*Z+k]);
            }
        }
    }

    return nom/den;

}

double m_c0_calc(py::array_t<double>& image
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
                nom += buf_image[k*Y*X+j*X+i]*(1- buf_u[i*Y*Z+j*Z+k]);
                den += (1-buf_u[i*Y*Z+j*Z+k]);
            }
        }
    }

    return nom/den;

}

py::array_t<double> m_abs_grad(py::array_t<int> u){

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

    int cnt = 0;
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
                //if ((cnt==0) and ((abs(xd) + abs(yd) + abs(zd))>0) ) {
                   // cnt++;
                    //std::cout << i << " " << j << " " << k << std::endl;
                //}
                resarr[k*Y*X+j*X+i] = (abs(xd) + abs(yd) + abs(zd) );
                //std::cout<<resarr[i*Y*Z + j*Z + k]<< std::endl;
            }

        }
    }
    //std::cout<<ptr1[89*Y*Z + 101*Z + 55]<<std::endl;
    result.resize({X,Y,Z});
    return result;
}


void m_modify_u_mat(py::array_t<double>& image,
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
                tmp = abs_ptr[k*Y*X+j*X+i] * (lambda1*(img_ptr[k*Y*X+j*X+i] - c1)*(img_ptr[k*Y*X+j*X+i] - c1) -
                                                  lambda2*(img_ptr[k*Y*X+j*X+i] - c0)*(img_ptr[k*Y*X+j*X+i] - c0)
                );
                if (tmp < 0) u_ptr[i*Y*Z+j*Z+k]=1;
                if (tmp > 0) u_ptr[i*Y*Z+j*Z+k]=0;

            }
        }
    }
    //std::cout<<u_ptr[89*Y*Z + 101*Z + 55]<<std::endl;
    //std::cout << "EXIT" << std::endl;
    u.resize({X,Y,Z});


}



py::array_t<int> m_morph_cv(py::array_t<double> image
        ,py::array_t<int> init_level_set,
                          int iterations,
                          int smoothing=1,double lambda1=1,double lambda2=1)
{
    py::module_ skim_sg = py::module_::import("skimage.segmentation.morphsnakes");
    //skim_sg.
    auto u = init_level_set;

    for (int l1 =0; l1 < iterations;l1++){

        double c0 = m_c0_calc(image,u);
        //std::cout << "c0"<< c0 << std::endl;
        double c1 = m_c1_calc(image,u);
        //std::cout << "c1" <<c1 << std::endl;
        auto abs_gr = m_abs_grad(u);
        //std::cout << "AG" << std::endl;
        m_modify_u_mat(image,abs_gr,u,lambda1,lambda2,c0,c1);
        //std::cout << "MMMMMMMMMMMM<<<<<<<<<<<<<<<<<<" << std::endl;
        for (int l2 =0; l2 < smoothing;l2++){

            u = skim_sg.attr("_curvop")(u);
            //std::cout << "sss<<<<<<<<<<<<<<<<<<" << std::endl;

        }



    }
    return u;


}





void compare_mats(std::vector<std::vector<std::vector<double>>> img1,py::array_t<double> im2)
{
    py::buffer_info buf1 = im2.request();

    long X = buf1.shape[0];
    long Y = buf1.shape[1];
    long Z = buf1.shape[2];
    std::cout << X<< " " << Y << " " << Z << std::endl;
    double * pt =(double *) buf1.ptr;
    for (int i = 0;i< X;i++) {

        for (int j = 0; j < Y; j++) {


            for (int k = 0; k < Z; k++) {
                //if (pt[k*Y*X+j*X+i] != img1[i][j][k])
                if (pt[i*Y*Z+j*Z+k] != img1[i][j][k])
                    std::cout << i << " " << j << " " << k << std::endl;
            }
        }
    }
    std::cout<< pt[46*Y*X+191*X+173] << " " << img1[173][191][46];
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

//m.def("add", &add, R"pbdoc(
//        Add two numbers
//        Some other explanation about the add function.
//    )pbdoc");

//m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
//        Subtract two numbers
//        Some other explanation about the subtract function.
//    )pbdoc");
    m.def("morph_cv", &m_morph_cv,"image"_a,"init_level_set"_a,"iterations"_a,"smoothing"_a=1,"lambda1"_a=1
            ,"lambda2"_a=1, R"pbdoc(
        C++ impl of morphological CV

    )pbdoc");

    m.def("compare_mats", &compare_mats, R"pbdoc(
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
