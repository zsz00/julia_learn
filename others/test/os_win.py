# -*- coding: utf-8 -*-
# python 3.x 获取windows系统信息
import wmi, time


class DISK:
    def __init__(self):
        self.device_id = ''
        self.disk_name = ''
        self.disk_caption = ''
        self.disk_size = 0
        self.partitions = []


class Partition:
    def __init__(self):
        self.device_id = ''
        self.partition_id = ''
        self.partition_name = ''
        self.partition_tatol = 0
        self.partition_free = 0
        self.partition_used = 0
        self.partition_used_Per = 0


class os_info():
    def __init__(self, show=False):
        self.show = show

    def sys_version(self):  # 获取操作系统版本
        c = wmi.WMI();
        os_1 = ''
        for sys in c.Win32_OperatingSystem():
            os_1 = "OS: %s, Bits: %s\n" % (sys.Caption, sys.OSArchitecture)
            if self.show:
                print(os_1)  # 系统版本和位数
                # print("Processes Num: %d" % sys.NumberOfProcesses)  # 当前系统运行的进程总数
        return os_1

    def cpu_mem(self):  # CPU类型和内存
        c = wmi.WMI()
        cpu_1 = ''
        mem_1 = 0
        for processor in c.Win32_Processor():
            cpu_1 += "%s: %s\n" % (processor.DeviceID, processor.Name.strip())
            # if self.show:
            #     print("%s: %s"%(processor.DeviceID,processor.Name.strip()))
        for Memory in c.Win32_PhysicalMemory():
            mem_1 += int(Memory.Capacity) / 1073741824
        mem_1 = "Memory: %.fGB\n" % mem_1

        if self.show:
            print(cpu_1)
            print(mem_1)
        return cpu_1, mem_1

    def cpu_use(self):
        # 5s取一次CPU的使用率
        c = wmi.WMI()
        while True:
            for cpu in c.Win32_Processor():
                timestamp = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
                print('%s | Utilization: %s: %d %%' % (timestamp, cpu.DeviceID, cpu.LoadPercentage))
                time.sleep(5)

    def disk(self):
        c = wmi.WMI()
        disk_info = []
        # 获取硬盘分区
        print("Disk:")
        for physical_disk in c.Win32_DiskDrive():
            disk1 = DISK()
            disk1.device_id = physical_disk.DeviceID
            disk1.disk_caption = physical_disk.Caption
            disk1.disk_size = int(physical_disk.Size)
            print(physical_disk.Caption, physical_disk.DeviceID, disk1.disk_size)  # 物理磁盘信息

            for partition in physical_disk.associators("Win32_DiskDriveToDiskPartition"):
                for logical_disk in partition.associators("Win32_LogicalDiskToPartition"):
                    partition1 = Partition()
                    partition1.device_id = physical_disk.DeviceID
                    partition1.partition_id = partition.Caption
                    partition1.partition_name = logical_disk.Caption
                #    print(physical_disk.Caption, partition.Caption,logical_disk.Caption)
                    for disk in c.Win32_LogicalDisk (DriveType=3):
                        if disk.Caption == logical_disk.Caption:    #获取硬盘分区信息
                            print("Caption: %s tatol: %0.1f G, free: %0.1f G, used_Per: %0.2f%% " %
                            (disk.Caption,int(disk.Size)/1024.0/1024/1024, int(disk.FreeSpace)/1024.0/1024/1024,
                            100.0 * (int(disk.Size)-int(disk.FreeSpace)) / int(disk.Size)))

                            partition1.partition_tatol = int(disk.Size)/1024.0/1024/1024
                            partition1.partition_free = int(disk.FreeSpace)/1024.0/1024/1024
                            partition1.partition_used = int(disk.Size)-int(disk.FreeSpace)/1024.0/1024/1024
                            partition1.partition_used_Per = 100.0*(int(disk.Size)-int(disk.FreeSpace))/int(disk.Size)
                            disk1.partitions.append(partition1)
            disk_info.append(disk1)
        return disk_info

    def network(self):
        c = wmi.WMI()
        print("Network:")
        # 获取MAC和IP地址
        for interface in c.Win32_NetworkAdapterConfiguration(IPEnabled=1):
            print("MAC: %s" % interface.MACAddress)
        for ip_address in interface.IPAddress:
            print("ip_add: %s" % ip_address)

    def process(self):
        c = wmi.WMI()
        # 获取自启动程序的位置
        for s in c.Win32_StartupCommand():
            print("[%s] %s <%s>" % (s.Location, s.Caption, s.Command))

        # 获取当前运行的进程
        for process in c.Win32_Process():
            print(process.ProcessId, process.Name)


def main():
    os = os_info(show=True)
    os_1 = os.sys_version()
    cpu_1, mem_1 = os.cpu_mem()
    disk_info = os.disk()
    os.network()
    # os.cpu_use()
    # os.process()


if __name__ == "__main__":
    main()
