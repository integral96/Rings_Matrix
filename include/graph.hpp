#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <type_traits>
#include <boost/hana.hpp>


#include "matrix_rings.hpp"

namespace hana = boost::hana;

template<char A, char B, bool predicat>
struct EDGE {
    static constexpr auto value = hana::make_pair(hana::make_pair(A, B), predicat);
};

template<typename ... Edges>
struct EDGE_LIST {};

template <class Edge>
struct GRAPH {
    static constexpr auto GRAPH_SET = hana::make_set(Edge::value);
};
template <class Edge, class ... Edges>
struct GRAPH<EDGE_LIST<Edge, Edges...>> {
    static constexpr auto GRAPH_SET = hana::insert(GRAPH<Edge>::GRAPH_SET, Edges::value...);
};

#endif // GRAPH_HPP
