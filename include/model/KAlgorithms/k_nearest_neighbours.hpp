#if !defined(AMT_K_ALGORITHMS_K_NEAREST_NEIGHBOURS_HPP)
#define AMT_K_ALGORITHMS_K_NEAREST_NEIGHBOURS_HPP

#include <dataframe.hpp>
#include <utils/distances.hpp>
#include <unordered_map>

namespace amt::classification{
    
    template<typename Distance = distance::Euclidean, typename... Ts>
    struct KNearestNeighbours{
        using frame_type = frame<Ts...>;
        constexpr KNearestNeighbours(KNearestNeighbours const& other) noexcept = default;
        constexpr KNearestNeighbours(KNearestNeighbours && other) noexcept = default;
        constexpr KNearestNeighbours& operator=(KNearestNeighbours const& other) noexcept = default;
        constexpr KNearestNeighbours& operator=(KNearestNeighbours && other) noexcept = default;
        ~KNearestNeighbours() = default;

        KNearestNeighbours(frame_type x, frame_type const& y, std::size_t neighbours = 2u, Distance dis = {}, std::size_t labels = 0u)
            : m_x(std::move(x))
            , m_y(std::move(y))
            , m_neighbours(neighbours)
            , m_dis(std::move(dis))
        {
            auto temp = unique(m_y[0]);
            m_labels = std::max( temp.size(), labels );
        }

        template<typename T, typename U>
        void print(std::vector< std::pair<T,U> > const& v){
            std::cout<<"[ ";
            for(auto const& [d,i] : v){
                std::cout<<"{ dist: "<< d<<", index: "<<i<<" }, ";
            }
            std::cout<<"]";
        }

        auto predict(frame_type const& x){
            auto xr = x.rows();
            
            frame_type ret(1u, xr);
            ret[0].name() = "Predicted Value";
            
            std::vector< std::pair<double, std::size_t> > dis;
            std::vector< std::size_t > count(m_labels,0);
            dis.resize(m_neighbours);
            

            for(auto i = 0u; i < xr; ++i){
                
                m_dis(dis, m_x, x, i);
                
                for(auto j = 0u; j < m_neighbours; ++j){
                    double el = m_y[0][dis[j].second];
                    ++count[ static_cast<std::size_t>(el) ];
                }

                auto it = std::max_element(count.begin(), count.end());
                auto it_dis = std::distance(std::begin(count), it);
            
                ret[0][i] = static_cast<double>(it_dis);
                std::fill(count.begin(), count.end(), 0u);
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

    // template<typename Distance, typename... Ts> 
    // KNearestNeighbours(frame<Ts...> const&, frame<Ts...> const&, std::size_t, Distance, std::size_t) -> KNearestNeighbours<Distance, Ts...>;

} // namespace amt::classification


#endif // AMT_K_ALGORITHMS_K_NEAREST_NEIGHBOURS_HPP
