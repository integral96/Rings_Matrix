#pragma once

#include <iostream>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <vector>
#include <future>

#include <boost/any.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/bind/bind.hpp>

#include "matrix_rings.hpp"

using namespace boost::placeholders;

template<typename T, size_t N>
std::array<T, N> operator + (const std::array<T, N>& A, const std::array<T, N>& B) {
    std::array<T, N> tmp;
    for(size_t i = 0; i < N; ++i ) tmp[i] = A[i] + B[i];
    return tmp;
}
template<typename T, size_t N>
boost::array<T, N> operator + (const boost::array<T, N>& A, const boost::array<T, N>& B) {
    boost::array<T, N> tmp;
    for(size_t i = 0; i < N; ++i ) tmp[i] = A[i] + B[i];
    return tmp;
}

template<size_t M, size_t N, typename T = double>
struct OMEGA {
protected:
    typedef T value_type;
    value_type H1, H2;
public:
    constexpr OMEGA(value_type l1, value_type l2) : H1(l1/M), H2(l2/N) { }
};

template<size_t M, size_t N, typename Func_G, typename Func_Phi>
class PUASSON_SOLVER : OMEGA<M, N> {
private:
    typedef OMEGA<M, N> super;
    typedef typename OMEGA<M, N>::value_type value_type;
    typedef typename boost::unordered_map<size_t, std::array<value_type, M>> vector_map;
    typedef typename boost::unordered_map<size_t, matrix_2<M, N, value_type>> matrix_map;
    typedef matrix_2<M, N, value_type> Matrix;

    vector_map YY_1;
    vector_map FF_1;

    Matrix CC;
    Matrix CC_invert;

    matrix_map ALPHA;
    vector_map BETA;
    Matrix EDIN;

    value_type H1, H2;

private:
    std::mutex m;
    std::vector<std::future<void>> tasks_alpha;
    std::vector<std::future<void>> tasks_beta;

private:
    Func_G   func_g;
    Func_Phi func_phi;
public:
    constexpr PUASSON_SOLVER(const Func_G& func_g_, const Func_Phi& func_phi_, value_type l1, value_type l2) :
        func_g(func_g_), func_phi(func_phi_), super(l1, l2), H1(super::H1*super::H1), H2(super::H2*super::H2)
    {
        value_type alpha = H2/H1;
        EDIN.identity_matrix();
        CC.three_diag_matrix(alpha);
        CC_invert = CC^CC;
        Border_conditions();
    }
    void output_alpha() {
        for(size_t i = 0; i < M + 1; i++) {
            SOLV_ALPHA(i);
        }
    }
    void output_beta() {
        for(size_t i = 0; i < M + 1; i++) {
            SOLV_BETA(i);
        }
    }
    void output_YY() {
        for(int i = M; i >= 0; --i ){
            SOLV_YY(i);
        }
    }
    void async_solv_alpha_beta() {
        for(size_t i = 0; i < M + 1; i++)
            tasks_alpha.push_back(std::async(std::launch::deferred, boost::bind(&PUASSON_SOLVER::SOLV_ALPHA, this, i)));
        for(auto& x: tasks_alpha) x.get();
        for(size_t i = 0; i < M + 1; i++)
            tasks_beta.push_back(std::async(std::launch::deferred, boost::bind(&PUASSON_SOLVER::SOLV_BETA, this, i)));
        for(auto& x: tasks_beta) x.get();
    }

    const std::array<value_type, M>& YY_(size_t i) {
        return YY_1[i];
    }
    const vector_map& YY_PRINT(size_t i) const {
        return YY_1;
    }
private:
    void SOLV_ALPHA(size_t k) {
        std::lock_guard<std::mutex> lk(m);

        if(k == 1) ALPHA[k] = CC_invert;
        ALPHA[k + 1] = ((CC - EDIN * ALPHA[k])^(CC - EDIN * ALPHA[k]))*EDIN;
    }
    void SOLV_BETA(size_t k) {
        std::lock_guard<std::mutex> lk(m);

        if(k == 1) BETA[k] = CC_invert*FF_1[0];
        BETA[k + 1] = ((CC - EDIN * ALPHA[k])^(CC - EDIN * ALPHA[k])) * (FF_1[k] + BETA[k]);
    }
    void SOLV_YY(size_t k) {
        if(k == M) YY_1[k] = BETA[k + 1];
        YY_1[k] = ALPHA[k + 1]*YY_1[k + 1] + BETA[k + 1];
    }

    void Border_conditions() {
        for(size_t j = 0; j < N + 1; ++j) {
            auto ptr_vec1 = std::make_unique<std::array<value_type, M>>();
            auto ptr_vec2 = std::make_unique<std::array<value_type, M>>();
            auto ptr_vec3 = std::make_unique<std::array<value_type, M>>();
            for(size_t i = 1; i < M; ++i) {
                if(i == 1) ptr_vec1->at(i) = (H2*(func_phi(super::H1*i, super::H2*j) +
                                                  1/H1*func_g(super::H1*(i - 1), super::H2*j)));
                else if(i == M - 1) ptr_vec1->at(i) = (H2*(func_phi(super::H1*i, super::H2*j) +
                                                           1/H1*func_g(super::H1*(i + 1), super::H2*j)));
                else ptr_vec1->at(i) = (H2*func_phi(super::H1*i, super::H2*j));
            }
            if(j == 0) {
                for(size_t i = 1; i < M; ++i) {
                    ptr_vec2->at(i) = (func_g(super::H1*i, super::H2*0));
                }
                FF_1.insert(std::make_pair(j, *ptr_vec2));
            }
            else if(j == N) {
                for(size_t i = 1; i < M; ++i) {
                    ptr_vec3->at(i) = (func_g(super::H1*i, super::H2*N));
                }
                FF_1.insert(std::make_pair(j, *ptr_vec3));
            }
            else FF_1.insert(std::make_pair(j, *ptr_vec1));
        }
    }

};

