#pragma once

#include <boost/mpl/bool.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/type_traits/enable_if.hpp>
#include <boost/multi_array.hpp>

#include <array>

#define BASE_FUNCTION_COUNT 10

///Factorial

template<size_t N>
struct factorial {
    static constexpr size_t value = N*factorial<N - 1>::value;
};
template<>
struct factorial<0> {
    static constexpr size_t value = 1;
};

///
///Вычисление степени
///

template<int N, typename T>
typename boost::enable_if_t<(N < 0), T> pow_(const T& x) {
    return T(1)/pow_<-N>(x);
}
template<int N, typename T>
typename boost::enable_if_t<(N == 0), T> pow_(const T& x) {
    return T(1);
}
template<int N, typename T>
typename boost::enable_if_t<(N > 0) && (N%2 == 0), T> pow_(const T& x) {
    T p = pow_<N / 2>(x);
    return p*p;
}
template<int N, typename T>
typename boost::enable_if_t<(N > 0) && (N%2 == 1), T> pow_(const T& x) {
    return pow_<N - 1>(x)*x;
}



/*!
 * struct meta_loop
 */
template <size_t N, size_t I, class Closure>
typename boost::enable_if_t<(I == N)> is_meta_loop(Closure& closure) {}

template <size_t N, size_t I, class Closure>
typename boost::enable_if_t<(I < N)> is_meta_loop(Closure& closure) {
    closure.template apply<I>();
    is_meta_loop<N, I + 1>(closure);
}
template <size_t N, class Closure>
void meta_loop(Closure& closure) {
    is_meta_loop<N, 0>(closure);
}
template <size_t N, class Closure>
void meta_loopUV(Closure& closure) {
    is_meta_loop<N, 1>(closure);
}
///++

/////Calculate Binom

template<size_t N, size_t K>
struct BC {
    static constexpr size_t value = factorial<N>::value / factorial<K>::value / factorial<N - K>::value;
};
/*!
 * struct abstract_sum
 */
template<class Closure>
struct abstract_sum_closures {
    typedef typename Closure::value_type value_type;
    abstract_sum_closures(Closure &closure) :  closure(closure), result(value_type()){}

    template<unsigned I>
    void apply(){
        result += closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};

template<unsigned N, class Closure>
typename Closure::value_type abstract_sums(Closure &closure) {
    abstract_sum_closures<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}

/*!
 * struct abstract_subtract
 */
template<class Closure>
struct abstract_subtract_closures {
    typedef typename Closure::value_type value_type;
    abstract_subtract_closures(Closure &closure) :  closure(closure), result(value_type()){}

    template<unsigned I>
    void apply(){
        result -= closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};

template<unsigned N, class Closure>
typename Closure::value_type abstract_subtract(Closure &closure) {
    abstract_subtract_closures<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}

/*!
 * struct abstract_divide
 */
template<class Closure>
struct abstract_divide_closures {
    typedef typename Closure::value_type value_type;
    abstract_divide_closures(Closure &closure) :  closure(closure), result(value_type()){}

    template<unsigned I>
    void apply(){
        result /= closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};

template<unsigned N, class Closure>
typename Closure::value_type abstract_divide(Closure &closure) {
    abstract_subtract_closures<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}
