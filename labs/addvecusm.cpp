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
        int *a = malloc_shared<int>(array_size, q);
        int *b = malloc_shared<int>(array_size, q);
        int *s = malloc_shared<int>(array_size, q);
        std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";
        {
            for (size_t i = 0; i < array_size; i++) {
                a[i] = i;
                b[i] = i;
            }
            range<1> num_items{array_size};
            q.submit([&](handler &h) {
                h.parallel_for(num_items, [=](auto i) { s[i] = a[i] + b[i]; });
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
    free(a, q);
    free(b, q);
    free(s, q);
    std::cout << "Vector add successfully completed on device.\n";
    return 0;
}
