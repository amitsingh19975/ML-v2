#if !defined(AMT_MODEL_K_ALGORITHMS_K_NEAREST_NEIGHBOURS_HPP)
#define AMT_MODEL_K_ALGORITHMS_K_NEAREST_NEIGHBOURS_HPP

#include <dataframe.hpp>
#include <utils/distances.hpp>
#include <unordered_map>

namespace amt::classification{
    
    template<typename Distance = distance::Euclidean, typename... Ts>
    struct KNearestNeighbours{
        using frame_type = frame;
        constexpr KNearestNeighbours(KNearestNeighbours const& other) noexcept = default;
        constexpr KNearestNeighbours(KNearestNeighbours && other) noexcept = default;
        constexpr KNearestNeighbours& operator=(KNearestNeighbours const& other) noexcept = default;
        constexpr KNearestNeighbours& operator=(KNearestNeighbours && other) noexcept = default;
        ~KNearestNeighbours() = default;

        KNearestNeighbours(frame_type x, frame_type y, std::size_t neighbours = 2u, Distance dis = {}, std::size_t labels = 0u)
            : m_x(std::move(x))
            , m_y(std::move(y))
            , m_neighbours(neighbours)
            , m_dis(std::move(dis))
        {
            if( m_x.rows() < neighbours ){
                throw std::length_error(
                    "amt::classification::KNearestNeighbours(frame_type, frame_type y, std::size_t, Distance, std::size_t) : "
                    "number of neighbours is greater than rows training data"
                );
            }
            auto temp = unique(m_y[0]);
            m_labels = std::max( temp.size(), labels );
        }

        auto predict(frame_type const& x){
            auto xr = x.rows();
            
            frame_type ret(1u, xr);
            ret[0].name() = "Predicted Value";
            
            
            #pragma omp parallel for schedule(static)
            for(auto i = 0u; i < xr; ++i){
                
                std::vector< std::pair<double, std::size_t> > dis(m_neighbours);
                std::vector< std::size_t > count(m_labels,0);
                
                m_dis(dis, m_x, x, i);
                
                for(auto j = 0u; j < m_neighbours; ++j){
                    double el = m_y[0][dis[j].second];
                    ++count[ static_cast<std::size_t>(el) ];
                }

                auto it = std::max_element(count.begin(), count.end());
                auto it_dis = std::distance(std::begin(count), it);
            
                ret[0][i] = static_cast<double>(it_dis);
            }
            return ret;
        }

    private:
        frame_type m_x;
        frame_type m_y;
        std::size_t m_neighbours{};
        std::size_t m_labels{};
        Distance m_dis;
    };

} // namespace amt::classification


#endif // AMT_MODEL_K_ALGORITHMS_K_NEAREST_NEIGHBOURS_HPP
