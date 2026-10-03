#!/bin/bash

ascii=$(cat ./intro.txt)
echo "$ascii"

echo "Welcome to the DROID setup process."

read -p "Is this your first time setting up the machine? (yes/no): " first_time

if [ "$first_time" = "yes" ]; then
        echo "Great! Let's proceed with the setup."

        echo "Repulling all submodules."
        read -p "Enter the user whose ssh credentials will be used: " USERNAME
        eval "$(ssh-agent -s)"
        ssh-add /home/$USERNAME/.ssh/id_ed25519
        ROOT_DIR="$(git rev-parse --show-toplevel)"
        cd $ROOT_DIR && git submodule update --recursive --remote --init

        echo -e "\nInstall docker \n"

        apt-get update
        apt-get install ca-certificates curl gnupg
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        chmod a+r /etc/apt/keyrings/docker.gpg
        echo \
          "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
          sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt-get update
        apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        systemctl enable docker

        echo -e "\nPerform realtime patch of kernel \n"

        apt update && apt install ubuntu-advantage-tools
        pro attach $UBUNTU_PRO_TOKEN
        pro enable realtime-kernel

        echo -e "\nSet cpu frequency scaling settings \n"

        apt install cpufrequtils -y
        systemctl disable ondemand
        systemctl enable cpufrequtils
        sh -c 'echo "GOVERNOR=performance" > /etc/default/cpufrequtils'
        systemctl daemon-reload && sudo systemctl restart cpufrequtils

else
    echo -e "\nWelcome back!\n"
fi

echo -e "\nSet environment variables from parameters file\n"

PARAMETERS_FILE="$(git rev-parse --show-toplevel)/droid/misc/parameters.py"
awk -F'[[:space:]]*=[[:space:]]*' '/^[[:space:]]*([[:alnum:]_]+)[[:space:]]*=/ && $1 != "ARUCO_DICT" { gsub("\"", "", $2); print "export " $1 "=" $2 }' "$PARAMETERS_FILE" > temp_env_vars.sh
source temp_env_vars.sh
export ROOT_DIR=$(git rev-parse --show-toplevel)
export NUC_IP=$nuc_ip
export ROBOT_IP=$robot_ip
export LAPTOP_IP=$laptop_ip
export SUDO_PASSWORD=$sudo_password
export ROBOT_TYPE=$robot_type
export ROBOT_SERIAL_NUMBER=$robot_serial_number
export HAND_CAMERA_ID=$hand_camera_id
export VARIED_CAMERA_1_ID=$varied_camera_1_id
export VARIED_CAMERA_2_ID=$varied_camera_2_id
export UBUNTU_PRO_TOKEN=$ubuntu_pro_token
rm temp_env_vars.sh

if [ "$ROBOT_TYPE" == "panda" ]; then
        export LIBFRANKA_VERSION=0.9.0
else
        export LIBFRANKA_VERSION=0.10.0
fi

read -p "Do you want to rebuild the container image? (yes/no): " first_time

if [ "$first_time" = "yes" ]; then
        echo -e "\n build control server container \n"

        DOCKER_COMPOSE_DIR="$ROOT_DIR/.docker/nuc"
        DOCKER_COMPOSE_FILE="$DOCKER_COMPOSE_DIR/docker-compose-nuc.yaml"
        cd $DOCKER_COMPOSE_DIR && docker-compose -f $DOCKER_COMPOSE_FILE build
fi

echo -e "\n set static ip \n"

echo "Select an Ethernet interface to set a static IP for:"

interfaces=$(ip -o link show | grep -Eo '^[0-9]+: (en|eth|ens|eno|enp)[a-z0-9]*' | awk -F' ' '{print $2}')

select interface_name in $interfaces; do
    if [ -n "$interface_name" ]; then
        break
    else
        echo "Invalid selection. Please choose a valid interface."
    fi
done

echo "You've selected: $interface_name"

nmcli connection delete "nuc_static"
nmcli connection add con-name "nuc_static" ifname "$interface_name" type ethernet
nmcli connection modify "nuc_static" ipv4.method manual ipv4.address $NUC_IP/24
nmcli connection up "nuc_static"

echo "Static IP configuration complete for interface $interface_name."

echo -e "8. run control server \n"

DOCKER_COMPOSE_FILE="$(git rev-parse --show-toplevel)/.docker/nuc/docker-compose-nuc.yaml"
docker compose -f $DOCKER_COMPOSE_FILE up -d
