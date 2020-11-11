#if !defined(AMT_UTILS_DISTANCES_HPP)
#define AMT_UTILS_DISTANCES_HPP

#include <cmath>
#include <tuple>
#include <dataframe.hpp>
#include <cassert>
#include <optional>

namespace amt::distance{

    template<typename T>
    struct is_pair : std::false_type{};

    template<typename T, typename U>
    struct is_pair< std::pair<T,U> > : std::true_type{};

    template<typename T>
    inline static constexpr bool is_pair_v = is_pair<T>::value;

    template<typename T>
    constexpr std::optional<std::size_t> find_pos(std::vector<T> const& v, double dis) noexcept {
        for(auto i = 0u; i < v.size(); ++i){
            double el = 0.;
            if constexpr( is_pair_v<T> ) el = v[i].first;
            else el = v[i];
            
            if( el > dis ) return {i};
        }
        return std::nullopt;
    }

    template<typename T>
    void shift_and_insert(std::vector<T>& v, double dis, T&& el){
        auto idx_opt = find_pos(v,dis);
        if(idx_opt){
            auto idx = static_cast<std::ptrdiff_t>(*idx_opt);
            auto sz = static_cast<std::ptrdiff_t>(v.size());
            auto last = sz - 1;
            for(auto k = sz - 2; k >= idx ; --k, --last){
                v[ static_cast<std::size_t>(last) ] = std::move(v[static_cast<std::size_t>(k)]);
            }
            v[static_cast<std::size_t>(idx)] = std::move(el);
        }
    }

    struct Euclidean{
        
        double operator()(double x1, double y1, double x2, double y2) const{
            auto dx = (x2 - x1);
            auto dy = (y2 - y1);
            return std::sqrt( (dx * dx) + (dy * dy) );
        }

        double operator()(Series auto const& vec1, Series auto const& vec2) const{
            assert(vec1.size() == vec2.size());
            auto sz = vec1.size();
            double dis = 0.;
            for(auto i = 0u; i < sz; ++i){
                double x1 = vec1[i];
                double x2 = vec2[i];
                auto dx = x2 -  x1;
                dis += dx * dx;
            }
            return std::sqrt(dis);
        }

        template<typename T>
        void operator()(std::vector< T >& dis_vec, Frame auto const& f1, Frame auto const& f2) const{
            using value_type = T;

            assert(f1.cols() == f2.cols());
            
            value_type def{};
            auto pinf = std::numeric_limits<double>::max();

            if constexpr( is_pair_v<value_type> ){
                def = { pinf, 0 };
            }else{
                def = pinf;
            }

            std::fill(dis_vec.begin(), dis_vec.end(), def);

            auto cols = f1.cols();
            auto rows = f1.rows();
            for(auto j = 0ul; j < rows; ++j){
                double dis = 0.;
                for(auto i = 0u; i < cols; ++i){
                    double x1 = f1[i][j];
                    double x2 = f2[i][j];
                    auto dx = x2 - x1;
                    dis += dx * dx;
                }

                auto temp = value_type{};
                
                dis = std::sqrt(dis);

                if constexpr( is_pair_v<value_type> ) temp = {dis, j};
                else temp = dis;

                shift_and_insert(dis_vec, dis, std::move(temp));
            }
        }

        
        template<typename T>
        void operator()(std::vector< T >& dis_vec, Frame auto const& f1, Frame auto const& f2, std::size_t f2_row) const{
            using value_type = T;

            assert(f1.cols() == f2.cols());
            
            value_type def{};
            auto pinf = std::numeric_limits<double>::max();

            if constexpr( is_pair_v<value_type> ){
                def = { pinf, 0 };
            }else{
                def = pinf;
            }

            std::fill(dis_vec.begin(), dis_vec.end(), def);

            auto cols = f1.cols();
            auto rows = f1.rows();
            for(auto j = 0ul; j < rows; ++j){
                double dis = 0.;
                for(auto i = 0u; i < cols; ++i){
                    double x1 = f1[i][j];
                    double x2 = f2[i][f2_row];
                    auto dx = x2 - x1;
                    dis += dx * dx;
                }

                auto temp = value_type{};
                
                dis = std::sqrt(dis);

                if constexpr( is_pair_v<value_type> ) temp = {dis, j};
                else temp = dis;
                
                shift_and_insert(dis_vec, dis, std::move(temp));
            }
        }

    };

    struct Manhattan{

        double operator()(double x1, double y1, double x2, double y2) const{
            auto dx = (x2 - x1);
            auto dy = (y2 - y1);
            return std::abs(dx) + std::abs(dy);
        }
        
        double operator()(Series auto const& vec1, Series auto const& vec2) const{
            assert(vec1.size() == vec2.size());
            auto sz = vec1.size();
            double dis = 0.;
            for(auto i = 0u; i < sz; ++i){
                double x1 = vec1[i];
                double x2 = vec2[i];
                auto dx = x2 - x1;
                dis += std::abs(dx);
            }
            return dis;
        }

        template<typename T>
        void operator()(std::vector< T >& dis_vec, Frame auto const& f1, Frame auto const& f2) const{
            using value_type = T;

            assert(f1.cols() == f2.cols());
            
            value_type def{};
            auto pinf = std::numeric_limits<double>::max();

            if constexpr( is_pair_v<value_type> ){
                def = { pinf, 0 };
            }else{
                def = pinf;
            }

            std::fill(dis_vec.begin(), dis_vec.end(), def);

            auto cols = f1.cols();
            auto rows = f1.rows();
            for(auto j = 0ul; j < rows; ++j){
                double dis = 0.;
                for(auto i = 0u; i < cols; ++i){
                    double x1 = f1[i][j];
                    double x2 = f2[i][j];
                    auto dx = x2 - x1;
                    dis += std::abs(dx);
                }

                auto temp = value_type{};

                if constexpr( is_pair_v<value_type> ) temp = {dis, j};
                else temp = dis;

                shift_and_insert(dis_vec, dis, std::move(temp));
            }
        }

        template<typename T>
        void operator()(std::vector< T >& dis_vec, Frame auto const& f1, Frame auto const& f2, std::size_t f2_row) const{
            using value_type = T;

            assert(f1.cols() == f2.cols());
            
            value_type def{};
            auto pinf = std::numeric_limits<double>::max();

            if constexpr( is_pair_v<value_type> ){
                def = { pinf, 0 };
            }else{
                def = pinf;
            }

            std::fill(dis_vec.begin(), dis_vec.end(), def);

            auto cols = f1.cols();
            auto rows = f1.rows();
            for(auto j = 0ul; j < rows; ++j){
                double dis = 0.;
                for(auto i = 0u; i < cols; ++i){
                    double x1 = f1[i][j];
                    double x2 = f2[i][f2_row];
                    auto dx = x2 - x1;
                    dis += std::abs(dx);
                }

                auto temp = value_type{};

                if constexpr( is_pair_v<value_type> ) temp = {dis, j};
                else temp = dis;

                shift_and_insert(dis_vec, dis, std::move(temp));
            }
        }

    };
    
    struct Minkowski{

        constexpr Minkowski() noexcept = default;
        constexpr Minkowski(Minkowski const& other) noexcept = default;
        constexpr Minkowski(Minkowski && other) noexcept = default;
        constexpr Minkowski& operator=(Minkowski const& other) noexcept = default;
        constexpr Minkowski& operator=(Minkowski && other) noexcept = default;
        ~Minkowski() = default;

        constexpr Minkowski(double p) noexcept
            : m_p(p)
        {}

        double operator()(double x1, double y1, double x2, double y2) const{
            double p_inv = 1 / m_p;
            auto dx = std::abs(x2 - x1);
            auto dy = std::abs(y2 - y1);
            auto dx_p = std::pow(dx,m_p);
            auto dy_p = std::pow(dy,m_p);
            return std::pow(dx_p + dy_p, p_inv);
        }

        double operator()(Series auto const& vec1, Series auto const& vec2) const{
            assert(vec1.size() == vec2.size());
            double p_inv = 1 / m_p;
            auto sz = vec1.size();
            double dis = 0.;
            for(auto i = 0u; i < sz; ++i){
                double x1 = vec1[i];
                double x2 = vec2[i];
                auto dx = std::pow( std::abs(x2 -  x1), m_p );
                dis += dx;
            }
            return std::pow(dis, p_inv);
        }

        template<typename T>
        void operator()(std::vector< T >& dis_vec, Frame auto const& f1, Frame auto const& f2) const{
            using value_type = T;

            double p_inv = 1 / m_p;
            assert(f1.cols() == f2.cols());
            
            value_type def{};
            auto pinf = std::numeric_limits<double>::max();

            if constexpr( is_pair_v<value_type> ){
                def = { pinf, 0 };
            }else{
                def = pinf;
            }

            std::fill(dis_vec.begin(), dis_vec.end(), def);

            auto cols = f1.cols();
            auto rows = f1.rows();
            for(auto j = 0ul; j < rows; ++j){
                double dis = 0.;
                for(auto i = 0u; i < cols; ++i){
                    double x1 = f1[i][j];
                    double x2 = f2[i][j];
                    auto dx = std::pow( std::abs(x2 -  x1), m_p );
                    dis += dx;
                }

                auto temp = value_type{};
                
                dis = std::pow(dis, p_inv);

                if constexpr( is_pair_v<value_type> ) temp = {dis, j};
                else temp = dis;

                shift_and_insert(dis_vec, dis, std::move(temp));
            }
        }

        template<typename T>
        void operator()(std::vector< T >& dis_vec, Frame auto const& f1, Frame auto const& f2, std::size_t f2_row) const{
            using value_type = T;

            double p_inv = 1 / m_p;
            assert(f1.cols() == f2.cols());
            
            value_type def{};
            auto pinf = std::numeric_limits<double>::max();

            if constexpr( is_pair_v<value_type> ){
                def = { pinf, 0 };
            }else{
                def = pinf;
            }

            std::fill(dis_vec.begin(), dis_vec.end(), def);

            auto cols = f1.cols();
            auto rows = f1.rows();
            for(auto j = 0ul; j < rows; ++j){
                double dis = 0.;
                for(auto i = 0u; i < cols; ++i){
                    double x1 = f1[i][j];
                    double x2 = f2[i][f2_row];
                    auto dx = std::pow( std::abs(x2 -  x1), m_p );
                    dis += dx;
                }

                auto temp = value_type{};
                
                dis = std::pow(dis, p_inv);

                if constexpr( is_pair_v<value_type> ) temp = {dis, j};
                else temp = dis;

                shift_and_insert(dis_vec, dis, std::move(temp));
            }
        }

    private:
        double m_p{2};
    };
    


} // namespace amt::distance


#endif // AMT_UTILS_DISTANCES_HPP
