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

#include "matrix.hpp"
#include "lagrange_solver.hpp"

using namespace boost::placeholders;

template<typename T, size_t N>
std::array<T, N> operator + (const std::array<T, N>& A, const std::array<T, N>& B) {
    std::array<T, N> tmp;
    for(size_t i = 0; i < N; ++i ) tmp[i] = A[i] + B[i];
    return tmp;
}

/// ===========================================
///
///
template<size_t M, size_t N, size_t L, typename Func_G, typename Func_Phi>
class PUASSON_SOLVER {
private:
    typedef double value_type;
    typedef typename boost::unordered_map<size_t, std::array<value_type, M>> vector_map;
    typedef typename boost::unordered_map<size_t, matrix_2<M, N, value_type>> matrix_map;
    typedef matrix_2<M, N, value_type> Matrix;

public:
    vector_map YY_1;
    vector_map FF_1;

    Matrix CC;
    Matrix CC_invert;

    matrix_map ALPHA;
    vector_map BETA;
    Matrix EDIN;

    value_type H11, H22;
    value_type alpha{};

private:
    std::vector<std::future<void>> tasks_alpha;
    std::vector<std::future<void>> tasks_beta;

    const LAGRANG_SET<N, L>& lagran_;

private:
    Func_G   func_g;
    Func_Phi func_phi;

private:
    struct SOLV_ALPHA  {
        SOLV_ALPHA(PUASSON_SOLVER<M, N, L, Func_G, Func_Phi>& solv) : solv(solv) {}
        template<size_t K>
        void apply(){
            if constexpr(K == 0) solv.ALPHA[K] = solv.CC_invert;
            solv.ALPHA[K + 1] = ((solv.CC - solv.EDIN * solv.ALPHA[K])^(solv.CC - solv.EDIN * solv.ALPHA[K]))*solv.EDIN;
        }
    private:
        PUASSON_SOLVER<M, N, L, Func_G, Func_Phi>& solv;
    };
    struct SOLV_BETA {
        SOLV_BETA(PUASSON_SOLVER<M, N, L, Func_G, Func_Phi>& solv) : solv(solv) {}
        template<size_t K>
        void apply(){
            if constexpr(K == 0) solv.BETA[K] = solv.CC_invert*solv.FF_1[0];
            solv.BETA[K + 1] = ((solv.CC - solv.EDIN * solv.ALPHA[K])^(solv.CC - solv.EDIN * solv.ALPHA[K])) *
                    (solv.FF_1[K] + solv.BETA[K]);
        }
    private:
        PUASSON_SOLVER<M, N, L, Func_G, Func_Phi>& solv;
    };
    struct SOLV_YY {
        SOLV_YY(PUASSON_SOLVER<M, N, L, Func_G, Func_Phi>& solv) : solv(solv) {}
        template<int K>
        void apply(){
            if(K == M) solv.YY_1[K] = solv.BETA[K + 1];
            solv.YY_1[K] = solv.ALPHA[K + 1]*solv.YY_1[K + 1] + solv.BETA[K + 1];
        }
    private:
        PUASSON_SOLVER<M, N, L, Func_G, Func_Phi>& solv;
    };
public:
    constexpr PUASSON_SOLVER(const Func_G& func_g_, const Func_Phi& func_phi_, const LAGRANG_SET<N, L>& lagran) :
        func_g(func_g_), func_phi(func_phi_), lagran_(lagran)
    {
        H11 = lagran_.get_step()*lagran_.get_step();// super::H1*super::H1;
        H22 = lagran_.get_step()*lagran_.get_step();// super::H2*super::H2;
        alpha = H22/H11;
        //
        EDIN.identity_matrix();
        CC.three_diag_matrix(alpha);
        CC_invert = CC^CC;
        //
        Border_conditions();
        //

        struct SOLV_ALPHA closure_alpha(*this);
        meta_loop<M + 1>(closure_alpha);
        struct SOLV_BETA closure_beta(*this);
        meta_loop<M + 1>(closure_beta);
        struct SOLV_YY closure_YY(*this);
        meta_loop_inv<M + 1>(closure_YY);
    }
    const std::array<value_type, M>& YY_(size_t i) {
        return YY_1[i];
    }
private:
    void Border_conditions() {
        for(size_t j = 0; j < N + 1; ++j) {
            auto ptr_vec1 = std::make_unique<std::array<value_type, M>>();
            auto ptr_vec2 = std::make_unique<std::array<value_type, M>>();
            auto ptr_vec3 = std::make_unique<std::array<value_type, M>>();
            for(size_t i = 1; i < M; ++i) {
                if(i == 1) ptr_vec1->at(i) = (H22*(func_phi(lagran_.get_XY()[i].first, lagran_.get_XY()[j].second) +
                                                  1/H11*func_g(lagran_.get_XY()[i - 1].first, lagran_.get_XY()[j].second)));
                else if(i == M - 1) ptr_vec1->at(i) = (H22*(func_phi(lagran_.get_XY()[i].first, lagran_.get_XY()[j].second) +
                                                           1/H11*func_g(lagran_.get_XY()[i + 1].first, lagran_.get_XY()[j].second)));
                else ptr_vec1->at(i) = (H22*func_phi(lagran_.get_XY()[i].first, lagran_.get_XY()[j].second));
            }
            if(j == 0) {
                for(size_t i = 1; i < M; ++i) {
                    ptr_vec2->at(i) = (func_g(lagran_.get_XY()[i].first, lagran_.get_y_min()));
                }
                FF_1.insert(std::make_pair(j, *ptr_vec2));
            }
            else if(j == N) {
                for(size_t i = 1; i < M; ++i) {
                    ptr_vec3->at(i) = (func_g(lagran_.get_XY()[i].first, lagran_.get_XY()[N -1].second));
                }
                FF_1.insert(std::make_pair(j, *ptr_vec3));
            }
            else FF_1.insert(std::make_pair(j, *ptr_vec1));
        }
    }

};
