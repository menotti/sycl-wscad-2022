#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <string>

#define array_size 10000

using namespace sycl;

int main(int argc, char* argv[]) {
    default_selector selector;
    try {
        queue q(selector);
        std::array<int, array_size> a, b, s; 
        std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";
        {
            buffer a_buf(a);
            buffer b_buf(b);
            buffer s_buf(s);        
            for (size_t i = 0; i < array_size; i++) {
                a[i] = i;
                b[i] = i;
            }
            range<1> num_items{array_size};
            q.submit([&](handler &h) {
                accessor aa(a_buf, h, read_only);
                accessor ab(b_buf, h, read_only);
                accessor as(s_buf, h, write_only, no_init);        
                h.parallel_for(num_items, [=](auto i) { as[i] = aa[i] + ab[i]; });
            }).wait();
        }
        int indices[]{0, 1, 2, (array_size - 2), (array_size - 1)};
        constexpr size_t indices_size = sizeof(indices) / sizeof(int);
        for (int i = 0; i < indices_size; i++) {
            int j = indices[i];
            if (i == indices_size - 2) 
                std::cout << "...\n";
            std::cout << "[" << j << "]: " << j << " + " << j << " = " << s[j] << "\n";
        }
    } catch (exception const &e) {
        std::cout << "An exception is caught while adding two vectors.\n";
        std::terminate();
    }
    std::cout << "Vector add successfully completed on device.\n";
    return 0;
}
