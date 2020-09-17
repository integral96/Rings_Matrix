
#include <iostream>
#include <fstream>

#include <GL/glut.h>

#include "include/matrix_rings.hpp"
#include "include/puason_equation.hpp"

float rot;
    float t{};

static auto func_g([](auto x, auto t) { return 45*x*(x - 1); });
static auto func_phi([](auto x, auto t) { return 0.235; });

static void key (int key, int x, int y)
{
    switch (key) {

    case GLUT_KEY_LEFT:
        rot = 2;
    break;

    case GLUT_KEY_RIGHT:
        rot = -2;
    break;

    case GLUT_KEY_UP:
    case GLUT_KEY_DOWN:
        rot = 0;
    break;
  }
  glutPostRedisplay();
}

void draw ()
{

    auto ptr_puasson = std::make_shared<PUASSON_SOLVER<BASE_FUNCTION_COUNT, BASE_FUNCTION_COUNT, decltype (func_g), decltype (func_phi)>>(
                                                             func_g, func_phi, 1., 1.);
    ptr_puasson->async_solv_alpha_beta();
    ptr_puasson->output_YY();
    glRotatef (rot, 0.5f, 0.5f, 0.0f);
    float x, z;
    glBegin (GL_LINES);
    for (size_t x = 0; x < BASE_FUNCTION_COUNT; x += 1)
    {
        for (size_t z = 0; z < BASE_FUNCTION_COUNT - 1; z += 1)
        {
            glVertex3f (x, t*ptr_puasson->YY_(x).at(z), z);
            glVertex3f (x+1, t*ptr_puasson->YY_(x + 1).at(z + 1), z+1);
        }
    }
    t += 0.003;
    glutPostRedisplay ();
    glEnd ();
}

void display ()
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw ();
    glutSwapBuffers ();
}

void init ()
{
    glEnable (GL_DEPTH_TEST);
    glMatrixMode (GL_PROJECTION);
    gluPerspective (15.0, 1.0, 1.0, 200.0);
    glMatrixMode (GL_MODELVIEW);
    gluLookAt( 0.0,   0.0, -100.0,
               2.0,   2.0,    2.0,
               1.0,   1.0,    0.0);

}


int main(int argc, char *argv[])
{
    //=====================================
    matrix_2<5, 5, double> AA{{1, 2, 3, 4,-2},
                                {-5, 5, 7, 8, 4},
                                {9, 10, -11, 12, 1},
                                {13, -14, -15, 0, 9},
                                {20, -26, 16, -17, 25}};
    matrix_2<5, 5, double> BB1{{1, 2, 3, 4,-2},
                                {-5, 5, 7, 8, 4},
                                {9, 10, -11, 12, 1},
                                {13, -14, -16, 0, 9},
                                {20, -26, 16, -17, 25}};
    std::array<double, 5> vector {1.5, 1.3, 2.5, 1.3, 3.2};
    std::cout << "AA = \n" << AA << std::endl;
    std::cout << "BB1 = \n" << BB1 << std::endl;
    std::cout << "AA * (AA + BB1) = \n" << AA * (AA + BB1) << std::endl;
    std::cout << "AA * (AA - BB1) = \n" << AA * (AA - BB1) << std::endl;
    std::cout << "AA^(-1) = \n" << (AA^AA) << std::endl;
    std::cout << "norma(AA) = " << calc_norma(AA) << std::endl;
    std::cout << "AA*vector = {";
    for(const auto& x : AA*vector) std::cout << x << "; ";
    std::cout << "}" << std::endl;
    std::cout << "AA/2*AA * (AA*5 + BB1/calc_norma(AA)) = \n" << AA/2*AA * (AA*5 + BB1/calc_norma(AA)) << std::endl;

    std::cout << "identity_matrix(AA) = \n" << AA.identity_matrix() << std::endl;
    //========================================================

    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glEnable(GL_DEPTH_TEST);
    glutCreateWindow ("function graph");
    glutDisplayFunc (display);
    glutSpecialFunc (key);


    init ();

    glutMainLoop ();
    return 0;
    /// ==================================================

}
