
#include <iostream>
#include <fstream>
#include <sstream>

#include "include/matrix_rings.hpp"
#include "include/graph.hpp"


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
    auto a = EDGE<'C', 'D', true>::value;
    auto b = hana::first(a);
    std::cout << hana::second(b) << std::endl;
    auto set = GRAPH<EDGE<'C', 'D', true>>::GRAPH_SET;
    std::stringstream out;
    hana::for_each(set, [&](auto arg) {
        out << hana::first(hana::first(arg)) <<" ";
    });
    std::cout << out.str() << std::endl;

    return 0;

}
