#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>

#include "bhy2.h"
#include "bhy2_parse.h"
#include "firmware/bhi260ap/BHI260AP.fw.h"

#define I2C_DEVICE "/dev/i2c-7"
#define I2C_ADDR   0x28

static int fd;
static struct bhy2_dev dev;
static int have_accel = 0, have_gyro = 0;
static int ax, ay, az, gx, gy, gz;

static int8_t read_i2c(uint8_t reg, uint8_t *data,
                       uint32_t len, void *unused)
{
    (void)unused;

    struct i2c_msg msgs[] = {
        { I2C_ADDR, 0, 1, &reg },
        { I2C_ADDR, I2C_M_RD, (uint16_t)len, data }
    };
    struct i2c_rdwr_ioctl_data t = { msgs, 2 };

    return ioctl(fd, I2C_RDWR, &t) < 0 ? -1 : 0;
}

static int8_t write_i2c(uint8_t reg, const uint8_t *data,
                        uint32_t len, void *unused)
{
    (void)unused;

    uint8_t buf[257];
    if (len > 256) return -1;

    buf[0] = reg;
    for (uint32_t i = 0; i < len; i++) buf[i + 1] = data[i];

    struct i2c_msg msg = {
        I2C_ADDR, 0, (uint16_t)(len + 1), buf
    };
    struct i2c_rdwr_ioctl_data t = { &msg, 1 };

    return ioctl(fd, I2C_RDWR, &t) < 0 ? -1 : 0;
}

static void delay_us(uint32_t us, void *unused)
{
    (void)unused;
    usleep(us);
}

static void check(int8_t rc, const char *where)
{
    if (rc != BHY2_OK) {
        fprintf(stderr, "%s failed: %d\n", where, rc);
        exit(1);
    }
}

static void callback(const struct bhy2_fifo_parse_data_info *info,
                     void *unused)
{
    (void)unused;

    struct bhy2_data_xyz v;
    bhy2_parse_xyz(info->data_ptr, &v);

    if (info->sensor_id == BHY2_SENSOR_ID_ACC_PASS) {
        ax = v.x;
        ay = v.y;
        az = v.z;
        have_accel = 1;
    } else if (info->sensor_id == BHY2_SENSOR_ID_GYRO_PASS) {
        gx = v.x;
        gy = v.y;
        gz = v.z;
        have_gyro = 1;
    }

    if (have_accel && have_gyro) {
        printf("ACC: %6d %6d %6d | GYRO: %5d %5d %5d\n",
               ax, ay, az, gx, gy, gz);
        fflush(stdout);
        have_accel = have_gyro = 0;
    }
}

int main(void)
{
    fd = open(I2C_DEVICE, O_RDWR);
    if (fd < 0) {
        perror("open I2C");
        return 1;
    }

    printf("Initializing BHI260AP...\n");
    fflush(stdout);

    check(bhy2_init(BHY2_I2C_INTERFACE, read_i2c, write_i2c,
                    delay_us, 256, NULL, &dev), "init");

    check(bhy2_soft_reset(&dev), "reset");

    printf("Uploading firmware...\n");
    fflush(stdout);

    check(bhy2_upload_firmware_to_ram(
              bhy2_firmware_image,
              sizeof(bhy2_firmware_image), &dev), "firmware upload");

    check(bhy2_boot_from_ram(&dev), "firmware boot");
    sleep(1);

    check(bhy2_register_fifo_parse_callback(
              BHY2_SENSOR_ID_ACC_PASS, callback, NULL, &dev),
          "acc callback");

    check(bhy2_register_fifo_parse_callback(
              BHY2_SENSOR_ID_GYRO_PASS, callback, NULL, &dev),
          "gyro callback");

    check(bhy2_update_virtual_sensor_list(&dev), "sensor list");

    check(bhy2_set_virt_sensor_cfg(
              BHY2_SENSOR_ID_ACC_PASS, 100.0f, 0, &dev),
          "accelerometer");

    check(bhy2_set_virt_sensor_cfg(
              BHY2_SENSOR_ID_GYRO_PASS, 100.0f, 0, &dev),
          "gyroscope");

    printf("Streaming acceleration and gyroscope. Ctrl+C to stop.\n");
    fflush(stdout);

    while (1) {
        uint8_t buffer[2048];

        check(bhy2_get_and_process_fifo(
                  buffer, sizeof(buffer), &dev), "FIFO");

        usleep(5000);
    }

    close(fd);
    return 0;
}