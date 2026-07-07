using CUDA
using LinearAlgebra

# Magnitud máxima de B por punto (Tesla). En puntos muy cerca de un imán el
# modelo dipolar diverge (1/r³); el clamp limita a |B| = B_MAX_T preservando
# dirección. 60 mT = 0.06 T.
const B_MAX_T = 0.06f0

# μ0 / 4π en SI (T·m/A)
const SCALE = 1.0f-7

function _Bnu!(R, P, M, B, n, m)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    Bx, By, Bz = 0.0f0, 0.0f0, 0.0f0

    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    @inbounds if i <= n
        Rx = R[i, 1]
        Ry = R[i, 2]
        Rz = R[i, 3]
    end

    @inbounds if i <= n
        # Loop principal: 4 dipolos por iteración para aumentar ILP.
        k = 1
        while k <= m - 3
            μx1, μy1, μz1 = M[1, k],   M[2, k],   M[3, k]
            px1, py1, pz1 = P[1, k],   P[2, k],   P[3, k]
            dx1, dy1, dz1 = Rx - px1, Ry - py1, Rz - pz1
            r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1

            μx2, μy2, μz2 = M[1, k+1], M[2, k+1], M[3, k+1]
            px2, py2, pz2 = P[1, k+1], P[2, k+1], P[3, k+1]
            dx2, dy2, dz2 = Rx - px2, Ry - py2, Rz - pz2
            r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2

            μx3, μy3, μz3 = M[1, k+2], M[2, k+2], M[3, k+2]
            px3, py3, pz3 = P[1, k+2], P[2, k+2], P[3, k+2]
            dx3, dy3, dz3 = Rx - px3, Ry - py3, Rz - pz3
            r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3

            μx4, μy4, μz4 = M[1, k+3], M[2, k+3], M[3, k+3]
            px4, py4, pz4 = P[1, k+3], P[2, k+3], P[3, k+3]
            dx4, dy4, dz4 = Rx - px4, Ry - py4, Rz - pz4
            r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4

            if r2_1 > 1.0f-8
                inv_r  = CUDA.rsqrt(r2_1)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx1*μx1 + dy1*μy1 + dz1*μz1
                Bx += 3.0f0 * dot_mr * dx1 * inv_r5 - μx1 * inv_r3
                By += 3.0f0 * dot_mr * dy1 * inv_r5 - μy1 * inv_r3
                Bz += 3.0f0 * dot_mr * dz1 * inv_r5 - μz1 * inv_r3
            end
            if r2_2 > 1.0f-8
                inv_r  = CUDA.rsqrt(r2_2)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx2*μx2 + dy2*μy2 + dz2*μz2
                Bx += 3.0f0 * dot_mr * dx2 * inv_r5 - μx2 * inv_r3
                By += 3.0f0 * dot_mr * dy2 * inv_r5 - μy2 * inv_r3
                Bz += 3.0f0 * dot_mr * dz2 * inv_r5 - μz2 * inv_r3
            end
            if r2_3 > 1.0f-8
                inv_r  = CUDA.rsqrt(r2_3)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx3*μx3 + dy3*μy3 + dz3*μz3
                Bx += 3.0f0 * dot_mr * dx3 * inv_r5 - μx3 * inv_r3
                By += 3.0f0 * dot_mr * dy3 * inv_r5 - μy3 * inv_r3
                Bz += 3.0f0 * dot_mr * dz3 * inv_r5 - μz3 * inv_r3
            end
            if r2_4 > 1.0f-8
                inv_r  = CUDA.rsqrt(r2_4)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx4*μx4 + dy4*μy4 + dz4*μz4
                Bx += 3.0f0 * dot_mr * dx4 * inv_r5 - μx4 * inv_r3
                By += 3.0f0 * dot_mr * dy4 * inv_r5 - μy4 * inv_r3
                Bz += 3.0f0 * dot_mr * dz4 * inv_r5 - μz4 * inv_r3
            end

            k += 4
        end

        # Cleanup: 0–3 dipolos restantes si m no es múltiplo de 4
        while k <= m
            μx, μy, μz = M[1, k], M[2, k], M[3, k]
            px, py, pz = P[1, k], P[2, k], P[3, k]
            dx, dy, dz = Rx - px, Ry - py, Rz - pz
            r2 = dx*dx + dy*dy + dz*dz

            if r2 > 1.0f-8
                inv_r  = CUDA.rsqrt(r2)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx*μx + dy*μy + dz*μz
                Bx += 3.0f0 * dot_mr * dx * inv_r5 - μx * inv_r3
                By += 3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3
                Bz += 3.0f0 * dot_mr * dz * inv_r5 - μz * inv_r3
            end
            k += 1
        end
    end

    if i <= n
        Bx *= SCALE
        By *= SCALE
        Bz *= SCALE
        # Clamp magnitud (preserva dirección) cerca de singularidades
        mag2 = Bx*Bx + By*By + Bz*Bz
        if mag2 > B_MAX_T * B_MAX_T
            s = B_MAX_T / sqrt(mag2)
            Bx *= s; By *= s; Bz *= s
        end
        @inbounds B[i, 1] = Bx
        @inbounds B[i, 2] = By
        @inbounds B[i, 3] = Bz
    end
    return
end
